#!/usr/bin/env python3
"""Materialize and verify the LGCVF plan in the existing DuckDB control plane.

The canonical ``FormalWorkPlan@1`` supplies semantic identities and dependency
structure.  The reviewed Markdown board supplies human-facing work metadata.
Both projections must agree before this trusted bootstrap writes anything.
After bootstrap, ``DatabaseTaskSource@1`` and ``DatabaseImplementationDaemon@1``
own operational state; this script never writes task status back to Markdown.

The configured profile is deliberately one-writer embedded DuckDB.  It does
not claim Quack qualification and does not install or probe network services.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import ast
import ctypes
import errno
import fcntl
import functools
import grp
import hashlib
import importlib.machinery
import importlib.metadata
import json
import os
import pwd
import select
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_RECOVERY_PYCACHE_CAPSULE_ATTRIBUTE = (
    "_lgcvf_isolated_recovery_pycache_capsule_v1"
)
_RECOVERY_PYCACHE_CAPSULE_SCHEMA = (
    "ipfs_accelerate_py/lgcvf-isolated-recovery-pycache@1"
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

ROOT = Path(__file__).resolve().parents[1]

_MAX_RECOVERY_IMPORT_ROOT_ENTRIES = 32_768
_MAX_RECOVERY_TRACKED_PATH_BYTES = 16 * 1024 * 1024
_MAX_RECOVERY_IMPORT_DEPTH = 64
_MAX_RECOVERY_IMPORT_FILE_BYTES = 32 * 1024 * 1024
_MAX_RECOVERY_IMPORT_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_RECOVERY_CONFIG_BYTES = 1024 * 1024
_RECOVERY_GIT_CONFIG_OVERRIDES = (
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
_RECOVERY_CONFIG_RELATIVE_PATH = (
    "config/agent_supervisor_logic_governed_compositional_verification_fabric_"
    "scheduler.json"
)
_RECOVERY_OMITTED_SOURCE_SYMLINKS = {
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
        logical = Path(value)
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
            logical = Path(value)
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


def _snapshot_head_bound_recovery_config(root: Path) -> dict[str, Any] | None:
    """Bind the canonical config to an ordinary index entry and exact HEAD bytes."""

    if _ISOLATED_RECOVERY_PYCACHE_DIRECTORY is None:
        return None
    relative = _RECOVERY_CONFIG_RELATIVE_PATH
    tracked = _bounded_recovery_git(
        root, "ls-files", "-v", "-z", "--cached", "--", relative
    )
    if tracked != f"H {relative}\0".encode():
        raise RuntimeError("protected recovery configuration has index flags")
    stage = _bounded_recovery_git(
        root, "ls-files", "-s", "-z", "--cached", "--", relative
    )
    committed = _bounded_recovery_git(
        root, "ls-tree", "-z", "HEAD", "--", relative
    )
    prefix = b"100644 blob "
    suffix = b"\t" + relative.encode("utf-8") + b"\0"
    if (
        not committed.startswith(prefix)
        or not committed.endswith(suffix)
        or len(committed) not in {len(prefix) + 40 + len(suffix), len(prefix) + 64 + len(suffix)}
    ):
        raise RuntimeError("protected recovery configuration HEAD entry differs")
    blob_oid = committed[len(prefix) : -len(suffix)]
    expected_stage = b"100644 " + blob_oid + b" 0" + suffix
    if stage != expected_stage:
        raise RuntimeError("protected recovery configuration index differs from HEAD")
    head_bytes = _bounded_recovery_git(
        root, "cat-file", "blob", f"HEAD:{relative}"
    )
    if not head_bytes or len(head_bytes) > _MAX_RECOVERY_CONFIG_BYTES:
        raise RuntimeError("protected recovery configuration bytes are unavailable")

    uid, private_gid = _private_recovery_import_principal()
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(root, directory_flags)
    config_fd = -1
    file_fd = -1
    try:
        config_fd = os.open("config", directory_flags, dir_fd=root_fd)
        file_fd = os.open(Path(relative).name, file_flags, dir_fd=config_fd)
        before = os.fstat(file_fd)
        mode = stat.S_IMODE(before.st_mode)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != uid
            or before.st_nlink != 1
            or mode & 0o002
            or (mode & 0o020 and before.st_gid != private_gid)
            or mode & 0o400 != 0o400
            or before.st_size < 1
            or before.st_size > _MAX_RECOVERY_CONFIG_BYTES
        ):
            raise RuntimeError("protected recovery configuration identity differs")
        chunks: list[bytes] = []
        observed = 0
        while True:
            block = os.read(
                file_fd,
                min(64 * 1024, _MAX_RECOVERY_CONFIG_BYTES + 1 - observed),
            )
            if not block:
                break
            chunks.append(block)
            observed += len(block)
            if observed > _MAX_RECOVERY_CONFIG_BYTES:
                raise RuntimeError("protected recovery configuration is too large")
        after = os.fstat(file_fd)
        worktree_bytes = b"".join(chunks)
        if (
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            or worktree_bytes != head_bytes
        ):
            raise RuntimeError("protected recovery configuration differs from HEAD")
        return {
            "relative_path": relative,
            "blob_oid": blob_oid.decode("ascii"),
            "sha256": "sha256:" + hashlib.sha256(head_bytes).hexdigest(),
            "bytes": head_bytes,
            "identity": (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_uid,
                before.st_gid,
                mode,
                before.st_nlink,
            ),
        }
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if config_fd >= 0:
            os.close(config_fd)
        os.close(root_fd)


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
        relative: Path,
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
            visit(root_fd, Path("."), 0)
        for name in (() if whole_repository else roots):
            directory_fd = os.open(name, directory_flags, dir_fd=root_fd)
            try:
                record(os.fstat(directory_fd), name, "directory")
                visit(directory_fd, Path(name), 0)
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
_ISOLATED_RECOVERY_CONFIG_AUTHORITY = _snapshot_head_bound_recovery_config(ROOT)
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

from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    read_coordination_history_projection,
    read_coordination_registry_projection,
)
from ipfs_accelerate_py.agent_supervisor.planning.formal_planning_contracts import (
    FormalWorkPlan,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import (
    CompletionAuthority,
    DeltaEffectClass,
    LifecycleState,
    MergeStrategyKind,
    PlanAuthorityRoots,
    PlanCompletionRule,
    PlanConflictContract,
    PlanDelta,
    PlanDeltaItem,
    PlanDeltaOperation,
    PlanLeaseContract,
    PlanMergeStrategy,
    PlanOrigin,
    PlanPopulationDigest,
    PlanProviderContract,
    PlanResourceContract,
    PlanRetryContract,
    PlanRevision,
    PlanWorktreeContract,
    PopulationKind,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    task_authority_spec_cid,
    task_projection_spec_cid,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.todo_vector_index import (
    parse_todo_blocks,
    split_csv,
)

CONFIG_PATH = ROOT / _RECOVERY_CONFIG_RELATIVE_PATH
FRESH_RECOVERY_TARGET_RELATIVE_ROOT = Path(
    "data/agent_supervisor/logic_governed_compositional_verification_fabric/run-v17"
)
SCHEMA = "ipfs_accelerate_py/agent-supervisor/lgcvf-duckdb-materialization@1"
VERIFICATION_SCHEMA = "ipfs_accelerate_py/agent-supervisor/lgcvf-duckdb-read-only-verification@1"
POPULATION_SCHEMA = "ipfs_accelerate_py/agent-supervisor/lgcvf-population@1"
SUCCESSOR_PREVIEW_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-successor-preview@1"
)
SUCCESSOR_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-successor-apply-receipt@1"
)
SUCCESSOR_VERIFICATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-successor-read-only-verification@1"
)
SUCCESSOR_COMPOSITE_PROJECTION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-successor-composite-projection@1"
)
SUCCESSOR_RECOVERY_MANIFEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-successor-recovery-manifest@1"
)
FRESH_RECOVERY_POLICY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-fresh-generation-recovery-policy@2"
)
FRESH_RECOVERY_PREVIEW_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-fresh-generation-recovery-preview@2"
)
FRESH_RECOVERY_MANIFEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-fresh-generation-recovery-manifest@3"
)
FRESH_RECOVERY_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-fresh-generation-recovery-receipt@3"
)
FRESH_RECOVERY_VERIFICATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-fresh-generation-recovery-verification@3"
)
FRESH_RECOVERED_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-recovered-evidence@1"
)
FRESH_RECOVERED_COMPLETION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-recovered-completion@1"
)
EXPECTED_NAMESPACE = "logic-governed-compositional-verification-fabric-v1"
EXPECTED_SCHEMA_REVISION = "datasets-authoritative-operational-v1"
EXPECTED_SCHEMA_PROFILE = "datasets-authoritative-operational"
SCHEMA_REVISION_ENV = "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION"
SUCCESSOR_ADDED_ALIAS = "LGCVF-113"
SUCCESSOR_AMENDED_ALIASES = frozenset(
    {
        "LGCVF-081",
        "LGCVF-111",
        "LGCVF-112",
        "LGCVF-120",
        "LGCVF-122",
        "LGCVF-124",
    }
)
SUCCESSOR_REPRIORITIZED_ALIASES = frozenset({"LGCVF-121", "LGCVF-123"})
SUCCESSOR_CHANGED_ALIASES = (
    SUCCESSOR_AMENDED_ALIASES | SUCCESSOR_REPRIORITIZED_ALIASES
)
SUCCESSOR_RUNTIME_DEPENDENCIES = {"LGCVF-120": (SUCCESSOR_ADDED_ALIAS,)}
FRESH_RECOVERY_CONSTRUCTION_COMPLETIONS = (
    "LGCVF-001",
    "LGCVF-002",
    "LGCVF-010",
    "LGCVF-020",
    "LGCVF-030",
    "LGCVF-040",
    "LGCVF-050",
)
FRESH_RECOVERY_MERGE_COMPLETIONS = (
    "LGCVF-051",
    "LGCVF-060",
    "LGCVF-061",
    "LGCVF-070",
    "LGCVF-071",
    "LGCVF-080",
)
FRESH_RECOVERY_REJECTED_SYNTHETIC = (
    "LGCVF-081",
    "LGCVF-090",
    "LGCVF-091",
    "LGCVF-100",
    "LGCVF-101",
    "LGCVF-102",
    "LGCVF-110",
    "LGCVF-111",
    "LGCVF-112",
    "LGCVF-113",
    "LGCVF-120",
    "LGCVF-122",
    "LGCVF-124",
)
FRESH_RECOVERY_PROTECTED_BLOCKERS = ("LGCVF-121", "LGCVF-123")
FRESH_RECOVERY_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "source_evidence_cid",
        "duckdb_runtime_cid",
        "source_generation",
        "target_generation",
        "source_runtime_root",
        "target_runtime_root",
        "source_head",
        "source_tree",
        "plan_root_cid",
        "population_root",
        "retained_completion_binding",
        "wrong_default_quarantine",
        "merge_completion_evidence",
        "validation_qualification",
        "validation_qualification_cid",
        "validation_projection_omission_commitment",
        "validation_projection_omission_root",
        "validation_projection_evidence_commitment",
        "validation_projection_evidence_root",
        "completion_partition",
        "synthetic_source_disposition",
        "source_database_statuses_read",
        "source_database_completion_records_imported",
        "model_provider_route",
        "network_isolation_enforced",
        "validation_cache_reused",
        "candidate_authored_validation",
        "validation_self_authority",
        "validation_completion_authoritative",
        "task_implementation_complete",
        "test_qualification_complete",
        "objective_complete",
        "release_qualified",
        "production_authorized",
        "authority",
        "manifest_cid",
    }
)
FRESH_RECOVERY_IMPORTED_COMPLETION_FIELDS = frozenset(
    {
        "task_id",
        "task_cid",
        "control_revision",
        "control_receipt_cid",
        "recovery_receipt_cid",
        "logical_completion_status",
        "validation_qualification_cid",
        "validation_observation_cid",
        "reconstruction_evidence_digest",
    }
)
FRESH_RECOVERY_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "source_generation",
        "target_generation",
        "source_head",
        "source_tree",
        "plan_root_cid",
        "population_root",
        "manifest_cid",
        "source_evidence_cid",
        "duckdb_runtime_cid",
        "bootstrap_receipt_cid",
        "bootstrap_receipt_sha256",
        "imported_completions",
        "completed_task_ids",
        "todo_task_ids",
        "blocked_task_ids",
        "completed_count",
        "todo_count",
        "blocked_count",
        "validation_qualification_cid",
        "validation_projection_omission_commitment",
        "validation_projection_omission_root",
        "validation_projection_evidence_commitment",
        "validation_projection_evidence_root",
        "model_provider_route",
        "network_isolation_enforced",
        "candidate_authored_validation",
        "validation_self_authority",
        "validation_completion_authoritative",
        "task_implementation_complete",
        "test_qualification_complete",
        "objective_complete",
        "release_qualified",
        "production_authorized",
        "source_database_statuses_read",
        "source_database_completion_records_imported",
        "synthetic_source_disposition",
        "operational_verification_root",
        "atomic_publish",
        "receipt_cid",
    }
)


class MaterializationError(RuntimeError):
    """Raised when bootstrap input or an operational store fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _strict_json_loads(value: str | bytes, *, noun: str) -> Any:
    """Decode JSON while rejecting duplicate object keys at every depth."""

    def closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = item
        return result

    try:
        return json.loads(value, object_pairs_hook=closed_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise MaterializationError(f"{noun} is invalid or ambiguous JSON") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _is_sha256(value: object) -> bool:
    """Return whether ``value`` is one closed, lowercase SHA-256 identity."""

    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _require_exact_fields(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], *, noun: str
) -> None:
    """Reject authority-bearing mappings whose closed schema has drifted."""

    if set(value) != set(expected):
        raise MaterializationError(f"{noun} fields differ")


def _safe_path(root: Path, value: Any, *, field: str) -> Path:
    text = str(value or "").strip()
    relative = Path(text)
    if not text or relative.is_absolute() or ".." in relative.parts:
        raise MaterializationError(f"{field} must be a safe repository-relative path")
    resolved = (root / relative).resolve(strict=False)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise MaterializationError(f"{field} escapes the repository root") from exc
    return resolved


def _lexical_safe_path(root: Path, value: Any, *, field: str) -> Path:
    """Return a confined path without resolving a potentially hostile link."""

    text = str(value or "").strip()
    relative = Path(text)
    if not text or relative.is_absolute() or ".." in relative.parts:
        raise MaterializationError(f"{field} must be a safe repository-relative path")
    resolved_root = root.resolve(strict=True)
    lexical = resolved_root.joinpath(*relative.parts)
    try:
        lexical.relative_to(resolved_root)
    except ValueError as exc:
        raise MaterializationError(f"{field} escapes the repository root") from exc
    return lexical


def _open_or_create_directory_chain(root: Path, directory: Path) -> int:
    """Open a confined directory through no-follow dirfds, creating parents."""

    resolved_root = root.resolve(strict=True)
    try:
        relative = directory.relative_to(resolved_root)
    except ValueError as exc:
        raise MaterializationError("fresh recovery target parent escapes root") from exc
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved_root, flags)
    try:
        for component in relative.parts:
            created = False
            try:
                os.mkdir(component, mode=0o755, dir_fd=descriptor)
                created = True
            except FileExistsError:
                pass
            if created:
                os.fsync(descriptor)
            child = os.open(component, flags, dir_fd=descriptor)
            status = os.fstat(child)
            if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.geteuid():
                os.close(child)
                raise MaterializationError(
                    "fresh recovery target parent identity differs"
                )
            if created:
                os.fsync(child)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _rename_noreplace(
    source: Path,
    destination_parent_fd: int,
    name: str,
    *,
    expected_source_identity: tuple[int, int],
) -> None:
    """Atomically publish with Linux RENAME_NOREPLACE or fail closed."""

    source_parent_fd = os.open(
        source.parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        source_status = os.stat(
            source.name, dir_fd=source_parent_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISDIR(source_status.st_mode)
            or (source_status.st_dev, source_status.st_ino)
            != expected_source_identity
        ):
            raise MaterializationError("fresh recovery stage identity changed")
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise MaterializationError("atomic no-replace publish is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            source_parent_fd,
            os.fsencode(source.name),
            destination_parent_fd,
            os.fsencode(name),
            1,  # RENAME_NOREPLACE
        )
        if result != 0:
            error = ctypes.get_errno()
            if error in {errno.EEXIST, errno.ENOTEMPTY}:
                raise MaterializationError(
                    "run-v17 collision detected during atomic publish"
                )
            raise MaterializationError(
                f"atomic no-replace publish failed: errno={error}"
            )
        os.fsync(source_parent_fd)
    finally:
        os.close(source_parent_fd)


def _fsync_tree(
    path: Path,
) -> tuple[tuple[int, int], dict[str, dict[str, Any]]]:
    """Durably flush and fingerprint a closed tree through no-follow FDs."""

    common_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags = (
        common_flags | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )
    root_descriptor = os.open(path, directory_flags)
    try:
        root_status = os.fstat(root_descriptor)
        if (
            not stat.S_ISDIR(root_status.st_mode)
            or root_status.st_uid != os.geteuid()
            or stat.S_IMODE(root_status.st_mode) & 0o022
            or stat.S_IMODE(root_status.st_mode) & 0o700 != 0o700
        ):
            raise MaterializationError("fresh recovery stage root identity differs")

        fingerprint: dict[str, dict[str, Any]] = {
            ".": {
                "kind": "directory",
                "mode": stat.S_IMODE(root_status.st_mode),
                "uid": root_status.st_uid,
                "dev": root_status.st_dev,
                "ino": root_status.st_ino,
            }
        }

        def flush(directory_descriptor: int, prefix: str) -> None:
            for name in sorted(os.listdir(directory_descriptor)):
                relative = f"{prefix}/{name}" if prefix else name
                before = os.stat(
                    name, dir_fd=directory_descriptor, follow_symlinks=False
                )
                if stat.S_ISDIR(before.st_mode):
                    child = os.open(name, directory_flags, dir_fd=directory_descriptor)
                    try:
                        after_open = os.fstat(child)
                        if (
                            after_open.st_uid != os.geteuid()
                            or stat.S_IMODE(after_open.st_mode) & 0o022
                            or stat.S_IMODE(after_open.st_mode) & 0o700 != 0o700
                            or (before.st_dev, before.st_ino)
                            != (after_open.st_dev, after_open.st_ino)
                        ):
                            raise MaterializationError(
                                "fresh recovery stage directory identity changed"
                            )
                        fingerprint[relative + "/"] = {
                            "kind": "directory",
                            "mode": stat.S_IMODE(after_open.st_mode),
                            "uid": after_open.st_uid,
                            "dev": after_open.st_dev,
                            "ino": after_open.st_ino,
                        }
                        flush(child, relative)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(before.st_mode):
                    child = os.open(
                        name,
                        common_flags | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=directory_descriptor,
                    )
                    try:
                        after_open = os.fstat(child)
                        if (
                            after_open.st_uid != os.geteuid()
                            or after_open.st_nlink != 1
                            or stat.S_IMODE(after_open.st_mode) & 0o022
                            or stat.S_IMODE(after_open.st_mode) & 0o400 != 0o400
                            or (before.st_dev, before.st_ino)
                            != (after_open.st_dev, after_open.st_ino)
                        ):
                            raise MaterializationError(
                                "fresh recovery stage file identity changed"
                            )
                        os.fsync(child)
                        chunks: list[bytes] = []
                        observed_size = 0
                        while True:
                            chunk = os.read(child, 1024 * 1024)
                            if not chunk:
                                break
                            observed_size += len(chunk)
                            if observed_size > 256 * 1024 * 1024:
                                raise MaterializationError(
                                    "fresh recovery stage file exceeds bound"
                                )
                            chunks.append(chunk)
                        after_read = os.fstat(child)
                        if (
                            (
                                after_open.st_dev,
                                after_open.st_ino,
                                after_open.st_size,
                                after_open.st_mtime_ns,
                            )
                            != (
                                after_read.st_dev,
                                after_read.st_ino,
                                after_read.st_size,
                                after_read.st_mtime_ns,
                            )
                            or observed_size != after_open.st_size
                        ):
                            raise MaterializationError(
                                "fresh recovery stage file changed while flushing"
                            )
                        fingerprint[relative] = {
                            "kind": "file",
                            "mode": stat.S_IMODE(after_open.st_mode),
                            "uid": after_open.st_uid,
                            "dev": after_open.st_dev,
                            "ino": after_open.st_ino,
                            "size": after_open.st_size,
                            "mtime_ns": after_open.st_mtime_ns,
                            "sha256": "sha256:"
                            + hashlib.sha256(b"".join(chunks)).hexdigest(),
                        }
                    finally:
                        os.close(child)
                else:
                    raise MaterializationError(
                        "fresh recovery stage contains a link or special file"
                    )
            os.fsync(directory_descriptor)

        flush(root_descriptor, "")
        after = os.fstat(root_descriptor)
        identity = (root_status.st_dev, root_status.st_ino)
        if (after.st_dev, after.st_ino) != identity:
            raise MaterializationError("fresh recovery stage root identity changed")
        return identity, fingerprint
    except OSError as exc:
        raise MaterializationError("fresh recovery stage cannot be flushed safely") from exc
    finally:
        os.close(root_descriptor)


def _seal_fresh_recovery_tree_permissions(path: Path) -> None:
    """Set the private staged authority tree to closed producer-owned modes."""

    common_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags = (
        common_flags | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )
    root_descriptor = os.open(path, directory_flags)
    try:
        root_status = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_status.st_mode) or root_status.st_uid != os.geteuid():
            raise MaterializationError("fresh recovery stage root identity differs")
        os.fchmod(root_descriptor, 0o700)
        count = 0

        def seal(directory_descriptor: int) -> None:
            nonlocal count
            for name in sorted(os.listdir(directory_descriptor)):
                count += 1
                if count > 10_000:
                    raise MaterializationError(
                        "fresh recovery stage population exceeds bound"
                    )
                before = os.stat(
                    name, dir_fd=directory_descriptor, follow_symlinks=False
                )
                if stat.S_ISDIR(before.st_mode):
                    child = os.open(
                        name, directory_flags, dir_fd=directory_descriptor
                    )
                    try:
                        opened = os.fstat(child)
                        if (
                            opened.st_uid != os.geteuid()
                            or (before.st_dev, before.st_ino)
                            != (opened.st_dev, opened.st_ino)
                        ):
                            raise MaterializationError(
                                "fresh recovery stage directory identity changed"
                            )
                        os.fchmod(child, 0o700)
                        seal(child)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(before.st_mode):
                    child = os.open(
                        name,
                        common_flags | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=directory_descriptor,
                    )
                    try:
                        opened = os.fstat(child)
                        if (
                            opened.st_uid != os.geteuid()
                            or opened.st_nlink != 1
                            or (before.st_dev, before.st_ino)
                            != (opened.st_dev, opened.st_ino)
                        ):
                            raise MaterializationError(
                                "fresh recovery stage file identity changed"
                            )
                        os.fchmod(child, 0o600)
                    finally:
                        os.close(child)
                else:
                    raise MaterializationError(
                        "fresh recovery stage contains a link or special file"
                    )

        seal(root_descriptor)
    except OSError as exc:
        raise MaterializationError(
            "fresh recovery stage permissions cannot be sealed"
        ) from exc
    finally:
        os.close(root_descriptor)


def _reconcile_stale_fresh_recovery_stages(staging_descriptor: int) -> int:
    """Remove only closed, producer-owned stale stage trees under the held lock."""

    # The stage root itself is required to remain exact 0700.  Inner paths can
    # retain pre-seal umask modes after an abrupt exit, so their safety comes
    # from that root containment plus exact owner/type/inode/link checks; only
    # special permission bits are rejected below.
    common_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags = (
        common_flags | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )

    def purge(directory_descriptor: int) -> None:
        for name in sorted(os.listdir(directory_descriptor)):
            before = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            mode = stat.S_IMODE(before.st_mode)
            if stat.S_ISDIR(before.st_mode):
                child = os.open(
                    name,
                    directory_flags,
                    dir_fd=directory_descriptor,
                )
                try:
                    opened = os.fstat(child)
                    opened_mode = stat.S_IMODE(opened.st_mode)
                    if (
                        not stat.S_ISDIR(opened.st_mode)
                        or opened.st_uid != os.geteuid()
                        or (opened.st_dev, opened.st_ino)
                        != (before.st_dev, before.st_ino)
                        or opened_mode != mode
                        or opened_mode & 0o7000
                        or opened_mode & 0o700 != 0o700
                    ):
                        raise MaterializationError(
                            "stale fresh recovery stage directory identity differs"
                        )
                    purge(child)
                    final = os.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                    if (final.st_dev, final.st_ino) != (
                        opened.st_dev,
                        opened.st_ino,
                    ) or stat.S_IMODE(final.st_mode) != opened_mode:
                        raise MaterializationError(
                            "stale fresh recovery stage directory identity changed"
                        )
                    os.rmdir(name, dir_fd=directory_descriptor)
                finally:
                    os.close(child)
            elif stat.S_ISREG(before.st_mode):
                child = os.open(
                    name,
                    common_flags | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_descriptor,
                )
                try:
                    opened = os.fstat(child)
                    opened_mode = stat.S_IMODE(opened.st_mode)
                    if (
                        not stat.S_ISREG(opened.st_mode)
                        or opened.st_uid != os.geteuid()
                        or opened.st_nlink != 1
                        or (opened.st_dev, opened.st_ino)
                        != (before.st_dev, before.st_ino)
                        or opened_mode != mode
                        or opened_mode & 0o7000
                    ):
                        raise MaterializationError(
                            "stale fresh recovery stage file identity differs"
                        )
                    final = os.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                    if (final.st_dev, final.st_ino) != (
                        opened.st_dev,
                        opened.st_ino,
                    ) or (
                        not stat.S_ISREG(final.st_mode)
                        or final.st_uid != opened.st_uid
                        or final.st_nlink != 1
                        or stat.S_IMODE(final.st_mode) != opened_mode
                    ):
                        raise MaterializationError(
                            "stale fresh recovery stage file identity changed"
                        )
                    os.unlink(name, dir_fd=directory_descriptor)
                finally:
                    os.close(child)
            else:
                raise MaterializationError(
                    "stale fresh recovery stage contains a link or special file"
                )
        os.fsync(directory_descriptor)

    reconciled = 0
    try:
        for name in sorted(os.listdir(staging_descriptor)):
            if name == "recovery.lock":
                continue
            if not name.startswith("stage-"):
                raise MaterializationError(
                    "fresh recovery staging directory contains an unknown entry"
                )
            before = os.stat(
                name,
                dir_fd=staging_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(before.st_mode):
                raise MaterializationError(
                    "stale fresh recovery stage root identity differs"
                )
            child = os.open(name, directory_flags, dir_fd=staging_descriptor)
            try:
                opened = os.fstat(child)
                mode = stat.S_IMODE(opened.st_mode)
                if (
                    opened.st_uid != os.geteuid()
                    or (opened.st_dev, opened.st_ino)
                    != (before.st_dev, before.st_ino)
                    or mode != 0o700
                ):
                    raise MaterializationError(
                        "stale fresh recovery stage root identity differs"
                    )
                purge(child)
                final = os.stat(
                    name,
                    dir_fd=staging_descriptor,
                    follow_symlinks=False,
                )
                if (final.st_dev, final.st_ino) != (
                    opened.st_dev,
                    opened.st_ino,
                ) or (
                    not stat.S_ISDIR(final.st_mode)
                    or final.st_uid != opened.st_uid
                    or stat.S_IMODE(final.st_mode) != mode
                ):
                    raise MaterializationError(
                        "stale fresh recovery stage root identity changed"
                    )
                os.rmdir(name, dir_fd=staging_descriptor)
            finally:
                os.close(child)
            os.fsync(staging_descriptor)
            reconciled += 1
    except OSError as exc:
        raise MaterializationError(
            "stale fresh recovery stage cannot be reconciled safely"
        ) from exc
    return reconciled


def _read_regular_evidence_bytes(
    root: Path,
    value: Any,
    *,
    field: str,
    expected_mode: str | None = None,
) -> tuple[Path, bytes, str]:
    """Read one evidence file once through a no-follow descriptor chain."""

    text = str(value or "").strip()
    relative = Path(text)
    lexical = _lexical_safe_path(root, text, field=field)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = flags | os.O_DIRECTORY
    descriptor = os.open(root.resolve(strict=True), directory_flags)
    try:
        for component in relative.parts[:-1]:
            child = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        try:
            file_descriptor = os.open(relative.name, flags, dir_fd=descriptor)
        except OSError as exc:
            raise MaterializationError(f"{field} is absent") from exc
        try:
            status = os.fstat(file_descriptor)
            if (
                not stat.S_ISREG(status.st_mode)
                or status.st_nlink != 1
                or status.st_uid != os.geteuid()
                or status.st_size > 64 * 1024 * 1024
                or (
                    expected_mode is not None
                    and oct(stat.S_IMODE(status.st_mode)) != expected_mode
                )
            ):
                raise MaterializationError(f"{field} file identity differs")
            chunks: list[bytes] = []
            observed = 0
            while True:
                chunk = os.read(file_descriptor, 1024 * 1024)
                if not chunk:
                    break
                observed += len(chunk)
                if observed > 64 * 1024 * 1024:
                    raise MaterializationError(f"{field} exceeds the evidence bound")
                chunks.append(chunk)
            data = b"".join(chunks)
            after = os.fstat(file_descriptor)
            if (
                (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or len(data) != status.st_size
            ):
                raise MaterializationError(f"{field} changed while being read")
        finally:
            os.close(file_descriptor)
    except OSError as exc:
        raise MaterializationError(f"{field} cannot be read safely") from exc
    finally:
        os.close(descriptor)
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    return lexical, data, digest


def _require_head_bound_recovery_bytes(
    root: Path,
    relative_value: Any,
    payload: bytes,
    *,
    field: str,
) -> tuple[str, str]:
    """Require raw authority bytes to equal one ordinary index/HEAD blob."""

    relative = Path(str(relative_value or ""))
    value = relative.as_posix()
    if (
        not value
        or relative.is_absolute()
        or ".." in relative.parts
        or value != str(relative_value)
    ):
        raise MaterializationError(f"{field} is not a canonical tracked path")
    try:
        entries = _tracked_recovery_import_entries(root, pathspecs=(value,))
    except RuntimeError as exc:
        raise MaterializationError(f"{field} Git authority differs") from exc
    entry = entries.get(value)
    if (
        len(entries) != 1
        or entry is None
        or entry[0] not in {"100644", "100755"}
        or not _git_blob_matches(payload, entry[1])
    ):
        raise MaterializationError(f"{field} raw bytes differ from index and HEAD")
    return entry


def _decode_evidence_json(data: bytes, *, noun: str) -> dict[str, Any]:
    value = _strict_json_loads(data, noun=noun)
    if not isinstance(value, dict):
        raise MaterializationError(f"{noun} must be an object")
    return value


def _snapshot_manifest_tree(
    root: Path,
    base_relative: Path,
    entries: Sequence[Mapping[str, Any]],
    *,
    noun: str,
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Snapshot a closed forensic tree through held no-follow directory FDs."""

    base = _lexical_safe_path(root, base_relative.as_posix(), field=noun)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(root.resolve(strict=True), flags)
    try:
        for component in base_relative.parts:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        status = os.fstat(descriptor)
        if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.geteuid():
            raise MaterializationError(f"{noun} root identity differs")
        expected: dict[str, Mapping[str, Any]] = {}
        children: dict[str, set[str]] = {"": set()}
        for raw in entries:
            relative = Path(str(raw.get("path") or ""))
            if (
                not relative.parts
                or relative.is_absolute()
                or ".." in relative.parts
                or relative.as_posix() in expected
            ):
                raise MaterializationError(f"{noun} entry path is unsafe")
            key = relative.as_posix()
            expected[key] = raw
            parent = relative.parent.as_posix()
            if parent == ".":
                parent = ""
            children.setdefault(parent, set()).add(relative.name)
            if str(raw.get("kind") or "") == "directory":
                children.setdefault(key, set())

        observed: list[dict[str, Any]] = []
        file_bytes: dict[str, bytes] = {}

        def walk(directory_fd: int, prefix: str) -> None:
            names = set(os.listdir(directory_fd))
            if names != children.get(prefix, set()):
                raise MaterializationError(f"{noun} entry closure differs")
            for name in sorted(names):
                key = f"{prefix}/{name}" if prefix else name
                raw = expected[key]
                item_status = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                kind = str(raw.get("kind") or "")
                mode = oct(stat.S_IMODE(item_status.st_mode))
                if kind == "directory":
                    if (
                        set(raw) != {"kind", "mode", "path"}
                        or not stat.S_ISDIR(item_status.st_mode)
                        or item_status.st_uid != os.geteuid()
                    ):
                        raise MaterializationError(f"{noun} directory identity differs")
                    child_fd = os.open(name, flags, dir_fd=directory_fd)
                    try:
                        opened = os.fstat(child_fd)
                        if (
                            not stat.S_ISDIR(opened.st_mode)
                            or opened.st_uid != os.geteuid()
                            or (opened.st_dev, opened.st_ino)
                            != (item_status.st_dev, item_status.st_ino)
                        ):
                            raise MaterializationError(
                                f"{noun} directory identity changed"
                            )
                        walk(child_fd, key)
                    finally:
                        os.close(child_fd)
                    item = {"kind": "directory", "mode": mode, "path": key}
                elif kind == "file":
                    if (
                        set(raw) != {"kind", "mode", "path", "sha256", "size"}
                        or not stat.S_ISREG(item_status.st_mode)
                        or item_status.st_uid != os.geteuid()
                        or item_status.st_nlink != 1
                    ):
                        raise MaterializationError(f"{noun} file identity differs")
                    file_fd = os.open(
                        name,
                        os.O_RDONLY
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=directory_fd,
                    )
                    try:
                        before = os.fstat(file_fd)
                        if (
                            not stat.S_ISREG(before.st_mode)
                            or before.st_uid != os.geteuid()
                            or before.st_nlink != 1
                            or (before.st_dev, before.st_ino)
                            != (item_status.st_dev, item_status.st_ino)
                        ):
                            raise MaterializationError(
                                f"{noun} file identity changed"
                            )
                        data_chunks: list[bytes] = []
                        while True:
                            chunk = os.read(file_fd, 1024 * 1024)
                            if not chunk:
                                break
                            data_chunks.append(chunk)
                        data = b"".join(data_chunks)
                        after = os.fstat(file_fd)
                    finally:
                        os.close(file_fd)
                    digest = "sha256:" + hashlib.sha256(data).hexdigest()
                    if (
                        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                        or len(data) != before.st_size
                        or len(data) != int(raw.get("size") or 0)
                        or digest != raw.get("sha256")
                    ):
                        raise MaterializationError(f"{noun} file content differs")
                    file_bytes[key] = data
                    item = {
                        "kind": "file",
                        "mode": mode,
                        "path": key,
                        "sha256": digest,
                        "size": len(data),
                    }
                elif kind == "symlink":
                    if (
                        set(raw) != {"kind", "mode", "path", "target"}
                        or not stat.S_ISLNK(item_status.st_mode)
                        or item_status.st_uid != os.geteuid()
                        or item_status.st_nlink != 1
                    ):
                        raise MaterializationError(f"{noun} symlink identity differs")
                    link_target = os.readlink(name, dir_fd=directory_fd)
                    after_link = os.stat(
                        name, dir_fd=directory_fd, follow_symlinks=False
                    )
                    if (
                        not stat.S_ISLNK(after_link.st_mode)
                        or (after_link.st_dev, after_link.st_ino)
                        != (item_status.st_dev, item_status.st_ino)
                    ):
                        raise MaterializationError(
                            f"{noun} symlink identity changed"
                        )
                    item = {
                        "kind": "symlink",
                        "mode": mode,
                        "path": key,
                        "target": link_target,
                    }
                else:
                    raise MaterializationError(f"{noun} entry kind differs")
                if item != dict(raw):
                    raise MaterializationError(f"{noun} entry identity differs")
                observed.append(item)

        walk(descriptor, "")
    except OSError as exc:
        raise MaterializationError(f"{noun} cannot be read safely") from exc
    finally:
        os.close(descriptor)
    del base
    return sorted(observed, key=lambda item: str(item["path"])), file_bytes


def _fresh_recovery_policy(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the closed run-generation recovery policy after coherence checks."""

    policy = config.get("fresh_generation_recovery")
    if not isinstance(policy, Mapping):
        raise MaterializationError("fresh_generation_recovery policy is required")
    expected_policy_fields = {
        "schema",
        "source_generation",
        "source_runtime_root",
        "target_generation",
        "target_runtime_root",
        "duckdb_runtime_cid",
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
    if set(policy) != expected_policy_fields:
        raise MaterializationError("fresh generation recovery policy fields differ")
    if policy.get("schema") != FRESH_RECOVERY_POLICY_SCHEMA:
        raise MaterializationError("fresh generation recovery policy schema differs")
    runtime_cid = str(policy.get("duckdb_runtime_cid") or "")
    if not runtime_cid.startswith("baguqeera"):
        raise MaterializationError("fresh recovery DuckDB runtime identity is absent")
    executable = str(policy.get("verification_python_executable") or "")
    executable_digest = str(
        policy.get("verification_python_executable_sha256") or ""
    )
    if not Path(executable).is_absolute() or not _is_sha256(executable_digest):
        raise MaterializationError("fresh recovery verification interpreter is absent")
    program = config.get("database_program")
    runtime = config.get("runtime_paths")
    if not isinstance(program, Mapping) or not isinstance(runtime, Mapping):
        raise MaterializationError("fresh recovery requires database and runtime paths")
    if set(runtime) != {
        "root",
        "state",
        "worktrees",
        "merge_queue",
        "logs",
        "evidence",
        "generated_runtime_artifacts_are_completion_authority",
    }:
        raise MaterializationError("fresh recovery runtime path fields differ")
    if set(program) != {
        "authoritative_records_schema_cas_and_fencing",
        "authoritative_transactional_data_model",
        "authority_mode",
        "event_store_path",
        "explicit_legacy",
        "export_profile",
        "failover_policy",
        "runtime_registry_path",
        "schema_profile",
        "schema_revision",
        "semantic_relations_permitted",
        "store_generation",
        "store_id",
        "task_source_kind",
        "worktree_root",
    }:
        raise MaterializationError("fresh recovery database program fields differ")
    source_generation = str(policy.get("source_generation") or "")
    target_generation = str(policy.get("target_generation") or "")
    source_root = str(policy.get("source_runtime_root") or "")
    target_root = str(policy.get("target_runtime_root") or "")
    if (
        source_generation != "lgcvf-run-v16"
        or target_generation != "lgcvf-run-v17"
        or source_generation == target_generation
        or source_root == target_root
        or not source_root.endswith("/run-v16")
        or not target_root.endswith("/run-v17")
    ):
        raise MaterializationError("fresh recovery generation boundary differs")
    if (
        runtime.get("root") != target_root
        or program.get("store_generation") != target_generation
        or program.get("export_profile") != target_generation
    ):
        raise MaterializationError("fresh recovery target generation is incoherent")
    target_prefix = target_root + "/"
    for field in ("state", "worktrees", "merge_queue", "logs", "evidence"):
        value = str(runtime.get(field) or "")
        if not value.startswith(target_prefix):
            raise MaterializationError(f"runtime_paths.{field} escapes run-v17")
    for field in (
        "store_id",
        "event_store_path",
        "runtime_registry_path",
        "worktree_root",
    ):
        value = str(program.get(field) or "")
        if not value.startswith(target_prefix):
            raise MaterializationError(f"database_program.{field} escapes run-v17")
    closed_lists = {
        "construction_completed_task_ids": FRESH_RECOVERY_CONSTRUCTION_COMPLETIONS,
        "recovered_completed_task_ids": FRESH_RECOVERY_MERGE_COMPLETIONS,
        "rejected_synthetic_task_ids": FRESH_RECOVERY_REJECTED_SYNTHETIC,
        "preserved_blocked_task_ids": FRESH_RECOVERY_PROTECTED_BLOCKERS,
    }
    for field, expected in closed_lists.items():
        observed = policy.get(field)
        if not isinstance(observed, list) or tuple(map(str, observed)) != expected:
            raise MaterializationError(f"fresh recovery {field} differs")
    completions = policy.get("merge_completions")
    merge_fields = {
        "task_id",
        "task_cid",
        "request_id",
        "completed_record_path",
        "completed_record_sha256",
        "train_receipt_path",
        "train_receipt_sha256",
        "candidate_commit",
        "merge_commit",
    }
    if not isinstance(completions, list) or tuple(
        str(item.get("task_id") or "")
        for item in completions
        if isinstance(item, Mapping)
    ) != FRESH_RECOVERY_MERGE_COMPLETIONS:
        raise MaterializationError("fresh recovery merge completion inventory differs")
    if len(completions) != len(FRESH_RECOVERY_MERGE_COMPLETIONS):
        raise MaterializationError("fresh recovery merge completion count differs")
    if any(not isinstance(item, Mapping) or set(item) != merge_fields for item in completions):
        raise MaterializationError("fresh recovery merge completion fields differ")
    source_prefix = source_root + "/"
    evidence_paths = (
        "retained_revision_receipt_path",
        "wrong_default_incident_manifest_path",
        "contaminated_coordination_projection_path",
    )
    if any(
        not str(policy.get(field) or "").startswith(source_prefix)
        for field in evidence_paths
    ):
        raise MaterializationError("fresh recovery evidence escapes run-v16")
    for item in completions:
        if not isinstance(item, Mapping) or any(
            not str(item.get(field) or "").startswith(source_prefix)
            for field in ("completed_record_path", "train_receipt_path")
        ):
            raise MaterializationError("fresh recovery merge evidence escapes run-v16")
    for field in (
        "retained_successor_revision_cid",
        "retained_delta_cid",
        "retained_completion_binding_cid",
        "retained_protected_blocker_binding_cid",
    ):
        if not str(policy.get(field) or "").startswith("b"):
            raise MaterializationError(f"fresh recovery {field} is absent")
    return policy


def _require_isolated_recovery_interpreter() -> None:
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
        raise MaterializationError(
            "protected recovery requires python -I -S -B"
        )
    _require_isolated_recovery_pycache()
    try:
        current_config_authority = _snapshot_head_bound_recovery_config(ROOT)
    except RuntimeError as exc:
        raise MaterializationError(
            "protected recovery configuration differs from HEAD"
        ) from exc
    if (
        _ISOLATED_RECOVERY_SOURCE_IDENTITY is None
        or _clean_recovery_import_source(ROOT)
        != _ISOLATED_RECOVERY_SOURCE_IDENTITY
        or _ISOLATED_RECOVERY_CONFIG_AUTHORITY is None
        or current_config_authority != _ISOLATED_RECOVERY_CONFIG_AUTHORITY
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
        raise MaterializationError("protected recovery import inventory differs")


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
        raise MaterializationError("protected recovery pycache isolation differs")
    try:
        observed_directory, observed_root, observed_identity = (
            _validated_isolated_recovery_pycache_capsule(capsule)
        )
    except RuntimeError as exc:
        raise MaterializationError(
            str(exc)
        ) from exc
    if (
        observed_directory is not _ISOLATED_RECOVERY_PYCACHE_DIRECTORY
        or observed_root != root
        or observed_identity != identity
    ):
        raise MaterializationError("protected recovery pycache isolation differs")


def _require_bound_duckdb_runtime_policy(config: Mapping[str, Any]) -> str:
    """Recompute the tracked DuckDB byte identity before recovery authority use."""

    _require_isolated_recovery_interpreter()
    policy = _fresh_recovery_policy(config)
    expected = str(policy["duckdb_runtime_cid"])
    try:
        executable = Path(sys.executable).resolve(strict=True)
    except OSError as exc:
        raise MaterializationError(
            "fresh recovery verification interpreter is unavailable"
        ) from exc
    if (
        str(executable) != policy["verification_python_executable"]
        or _sha256_file(executable)
        != policy["verification_python_executable_sha256"]
    ):
        raise MaterializationError(
            "fresh recovery verification interpreter differs from configuration"
        )
    try:
        from scripts.qualify_logic_governed_compositional_verification_fabric import (
            QualificationError,
            bound_duckdb_runtime_evidence,
        )

        evidence = bound_duckdb_runtime_evidence()
    except (ImportError, OSError, QualificationError, TypeError, ValueError) as exc:
        raise MaterializationError("bound DuckDB runtime is unavailable") from exc
    if evidence.get("runtime_cid") != expected:
        raise MaterializationError("bound DuckDB runtime differs from configuration")
    return expected


def _with_bound_duckdb_runtime(
    function: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Execute one recovery operation solely through the pinned runtime capsule."""

    @functools.wraps(function)
    def wrapped(config: Mapping[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        expected = _require_bound_duckdb_runtime_policy(config)
        try:
            from scripts.qualify_logic_governed_compositional_verification_fabric import (
                QualificationError,
                isolated_bound_duckdb_runtime,
            )
        except (ImportError, OSError, QualificationError, TypeError, ValueError) as exc:
            raise MaterializationError("bound DuckDB runtime admission failed") from exc
        runtime = isolated_bound_duckdb_runtime(expected_runtime_cid=expected)
        try:
            runtime.__enter__()
        except (ImportError, OSError, QualificationError, TypeError, ValueError) as exc:
            raise MaterializationError("bound DuckDB runtime admission failed") from exc
        try:
            result = function(config, *args, **kwargs)
        except BaseException:
            try:
                runtime.__exit__(*sys.exc_info())
            except (ImportError, OSError, QualificationError, TypeError, ValueError) as exc:
                raise MaterializationError(
                    "bound DuckDB runtime validation failed"
                ) from exc
            raise
        try:
            runtime.__exit__(None, None, None)
        except (ImportError, OSError, QualificationError, TypeError, ValueError) as exc:
            raise MaterializationError("bound DuckDB runtime validation failed") from exc
        return result

    return wrapped


def _targets_fresh_recovery_generation(
    config: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> bool:
    """Recognize the protected generation through markers or resolved paths."""

    if "fresh_generation_recovery" in config:
        return True
    program = config.get("database_program")
    runtime = config.get("runtime_paths")
    values: list[str] = []
    path_values: list[str] = []
    if isinstance(program, Mapping):
        values.extend(
            str(program.get(field) or "")
            for field in (
                "store_generation",
                "export_profile",
                "store_id",
                "event_store_path",
                "runtime_registry_path",
                "worktree_root",
            )
        )
        path_values.extend(
            str(program.get(field) or "")
            for field in (
                "store_id",
                "event_store_path",
                "runtime_registry_path",
                "worktree_root",
            )
        )
    if isinstance(runtime, Mapping):
        values.extend(str(value or "") for value in runtime.values())
        path_values.extend(
            str(runtime.get(field) or "")
            for field in (
                "root",
                "state",
                "worktrees",
                "merge_queue",
                "logs",
                "evidence",
            )
        )
    if any(
        value == "lgcvf-run-v17"
        or value == "logic-governed-compositional-verification-fabric-run-v17"
        or "/run-v17" in value.replace("\\", "/")
        for value in values
    ):
        return True
    try:
        resolved_root = root.resolve(strict=False)
    except OSError:
        return True
    protected_lexical = resolved_root / FRESH_RECOVERY_TARGET_RELATIVE_ROOT
    protected_resolved = protected_lexical.resolve(strict=False)
    for text in path_values:
        if not text:
            continue
        raw = Path(text)
        lexical = raw if raw.is_absolute() else resolved_root / raw
        try:
            resolved = lexical.resolve(strict=False)
        except OSError:
            continue
        if (
            lexical == protected_lexical
            or lexical.is_relative_to(protected_lexical)
            or resolved == protected_resolved
            or resolved.is_relative_to(protected_resolved)
        ):
            return True
    return False


def load_config(
    config_path: Path = CONFIG_PATH,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Load the closed embedded profile without importing any provider."""

    try:
        if (
            config_path == CONFIG_PATH
            and _ISOLATED_RECOVERY_CONFIG_AUTHORITY is not None
        ):
            raw = _ISOLATED_RECOVERY_CONFIG_AUTHORITY["bytes"]
        else:
            raw = config_path.read_bytes()
        payload = _strict_json_loads(
            raw, noun="scheduler config"
        )
    except (OSError, MaterializationError) as exc:
        raise MaterializationError(f"scheduler config is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise MaterializationError("scheduler config root must be an object")
    if payload.get("board_namespace") != EXPECTED_NAMESPACE:
        raise MaterializationError("unexpected LGCVF board namespace")
    program = payload.get("database_program")
    if not isinstance(program, dict):
        raise MaterializationError("database_program is required")
    expected = {
        "authority_mode": "embedded",
        "task_source_kind": "duckdb",
        "schema_revision": EXPECTED_SCHEMA_REVISION,
        "schema_profile": EXPECTED_SCHEMA_PROFILE,
        "semantic_relations_permitted": False,
        "failover_policy": "fail_closed",
    }
    for field, value in expected.items():
        if program.get(field) != value:
            raise MaterializationError(f"database_program.{field} must equal {value!r}")
    writer = payload.get("bootstrap_writer_policy")
    if not isinstance(writer, dict):
        raise MaterializationError("bootstrap_writer_policy is required")
    if (
        writer.get("maximum_processes") != 1
        or writer.get("direct_multi_process_duckdb_permitted") is not False
        or writer.get("automatic_installation_permitted") is not False
    ):
        raise MaterializationError("LGCVF bootstrap must remain one-writer and offline")
    if int(payload.get("max_lanes") or 0) != 1:
        raise MaterializationError("embedded LGCVF authority permits exactly one lane")
    _fresh_recovery_policy(payload)
    for field in (
        "taskboard_path",
        "objectives_path",
        "plan_path",
        "formal_plan_path",
        "validator_path",
        "materializer_path",
    ):
        path = _safe_path(root, payload.get(field), field=field)
        if not path.is_file():
            raise MaterializationError(f"required LGCVF source is absent: {field}")
    _safe_path(root, program.get("store_id"), field="database_program.store_id")
    return payload


def _git(root: Path, *argv: str) -> str:
    try:
        substitution_before = _git_object_substitution_state(root)
    except RuntimeError as exc:
        raise MaterializationError(
            "protected recovery Git object substitution differs"
        ) from exc
    completed = subprocess.run(
        [
            "/usr/bin/git",
            *_RECOVERY_GIT_CONFIG_OVERRIDES,
            "-c",
            "core.hooksPath=/dev/null",
            *argv,
        ],
        cwd=root,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LANG": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        },
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise MaterializationError(f"git {' '.join(argv)} failed: {detail}")
    try:
        if _git_object_substitution_state(root) != substitution_before:
            raise MaterializationError(
                "protected recovery Git object substitution changed"
            )
    except RuntimeError as exc:
        raise MaterializationError(
            "protected recovery Git object substitution differs"
        ) from exc
    return completed.stdout.strip()


def _require_git_ignored_recovery_paths(
    source_root: Path,
    paths: Sequence[Path],
) -> None:
    """Require every recovery scratch path inside source to be Git-ignored."""

    resolved_source = source_root.resolve(strict=True)
    for path in paths:
        try:
            relative = path.relative_to(resolved_source)
        except ValueError:
            continue
        try:
            _git(
                resolved_source,
                "check-ignore",
                "-q",
                "--",
                relative.as_posix(),
            )
        except MaterializationError as exc:
            raise MaterializationError(
                "fresh recovery staging path is not Git-ignored"
            ) from exc


def _project_source_binding(
    config: Mapping[str, Any],
    *,
    root: Path,
    require_clean: bool,
) -> dict[str, Any]:
    """Project exact repository topology, optionally requiring clean worktrees."""

    accelerator_dirty = bool(
        _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    )
    if require_clean and accelerator_dirty:
        raise MaterializationError(
            "refusing to materialize from a dirty execution worktree"
        )
    binding = config.get("source_binding")
    if not isinstance(binding, Mapping):
        raise MaterializationError("source_binding must be an object")
    branch = _git(root, "symbolic-ref", "--short", "HEAD")
    expected_branch = str(binding.get("accelerator_required_branch") or "")
    if branch != expected_branch:
        raise MaterializationError(
            f"accelerator branch differs: expected {expected_branch!r}, observed {branch!r}"
        )
    head = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    ancestor = str(binding.get("accelerator_required_ancestor") or "")
    _git(root, "merge-base", "--is-ancestor", ancestor, head)

    datasets = _safe_path(
        root,
        binding.get("ipfs_datasets_submodule_path"),
        field="source_binding.ipfs_datasets_submodule_path",
    )
    datasets_dirty = bool(
        _git(datasets, "status", "--porcelain=v1", "--untracked-files=all")
    )
    if require_clean and datasets_dirty:
        raise MaterializationError("ipfs_datasets_py nested worktree is dirty")
    datasets_head = _git(datasets, "rev-parse", "HEAD")
    datasets_tree = _git(datasets, "rev-parse", "HEAD^{tree}")
    expected_datasets = str(binding.get("ipfs_datasets_planning_revision") or "")
    if datasets_head != expected_datasets:
        raise MaterializationError("ipfs_datasets_py HEAD differs from the configured revision")
    relative = datasets.relative_to(root).as_posix()
    gitlink = _git(root, "ls-tree", head, "--", relative).split()
    if (
        len(gitlink) < 3
        or gitlink[0] != "160000"
        or gitlink[1] != "commit"
        or gitlink[2] != datasets_head
    ):
        raise MaterializationError("ipfs_datasets_py is not the exact configured gitlink")
    report = {
        "accelerator_branch": branch,
        "accelerator_head": head,
        "accelerator_tree": tree,
        "accelerator_required_ancestor": ancestor,
        "datasets_gitlink": datasets_head,
        "datasets_head": datasets_head,
        "datasets_tree": datasets_tree,
        "datasets_path": relative,
        "nested_repository_count": 1,
        "worktrees_clean": not accelerator_dirty and not datasets_dirty,
    }
    report["source_forest_root"] = content_identity(report)
    return report


def verify_source_binding(config: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    """Bind bootstrap to one clean branch, tree, and exact datasets gitlink."""

    return _project_source_binding(config, root=root, require_clean=True)


def _metadata_bool(fields: Mapping[str, str], key: str) -> bool:
    value = str(fields.get(key) or "").strip().casefold()
    if value not in {"true", "false"}:
        raise MaterializationError(f"Markdown field {key!r} must be true or false")
    return value == "true"


def project_population(
    config: Mapping[str, Any],
    *,
    formal_plan: FormalWorkPlan,
    todo_text: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a checked DatabaseTaskSource population from both plan views."""

    namespace = str(formal_plan.metadata.get("board_namespace") or "")
    if namespace != EXPECTED_NAMESPACE:
        raise MaterializationError("formal plan namespace differs from the scheduler")
    logical_task_prefix = formal_plan.metadata.get("task_prefix")
    scheduler_task_prefix = config.get("task_prefix")
    if (
        not isinstance(logical_task_prefix, str)
        or not isinstance(scheduler_task_prefix, str)
        or scheduler_task_prefix != "## " + logical_task_prefix
    ):
        raise MaterializationError(
            "scheduler Markdown task selector differs from the formal logical prefix"
        )
    plan_binding = config.get("plan_binding")
    if not isinstance(plan_binding, Mapping):
        raise MaterializationError("plan_binding must be an object")
    if formal_plan.content_id != str(plan_binding.get("formal_plan_content_id") or ""):
        raise MaterializationError("formal plan content identity differs from the scheduler")
    if formal_plan.metadata.get("predecessor_plan_cid") != plan_binding.get("predecessor_plan_cid"):
        raise MaterializationError("formal plan predecessor identity differs")

    blocks = parse_todo_blocks(todo_text, task_header_prefix=scheduler_task_prefix)
    block_map = {task_id: (title, line, fields) for task_id, title, line, fields in blocks}
    formal_ids = tuple(task.task_id for task in formal_plan.tasks)
    if tuple(block_map) != formal_ids:
        raise MaterializationError("Markdown task order/identity differs from FormalWorkPlan")

    root_goal = formal_plan.goals[0]
    goal_cids = {root_goal.goal_id: root_goal.content_id}
    goal_records: list[dict[str, Any]] = [
        {
            "goal_cid": root_goal.content_id,
            "goal_id": root_goal.goal_id,
            "goal_alias": root_goal.goal_id,
            "title": str(root_goal.metadata.get("title") or root_goal.goal_id),
            "ordinal": 1,
            "status": "open",
            "objective_id": "objective:lgcvf-root",
            "objective_alias": root_goal.goal_id,
            "priority": "P0",
            "formal_content_id": root_goal.content_id,
            "formal_record": root_goal.to_dict(),
        }
    ]
    for ordinal, subgoal in enumerate(formal_plan.subgoals, start=2):
        goal_cids[subgoal.subgoal_id] = subgoal.content_id
        goal_records.append(
            {
                "goal_cid": subgoal.content_id,
                "goal_id": subgoal.subgoal_id,
                "goal_alias": subgoal.subgoal_id,
                "title": str(subgoal.metadata.get("title") or subgoal.subgoal_id),
                "ordinal": ordinal,
                "status": "open",
                "parent_goal_cid": root_goal.content_id,
                "priority": "P0",
                "formal_content_id": subgoal.content_id,
                "formal_record": subgoal.to_dict(),
            }
        )

    goal_edges: list[dict[str, str]] = []
    for subgoal in formal_plan.subgoals:
        goal_edges.append(
            {
                "parent_goal_cid": root_goal.content_id,
                "child_goal_cid": subgoal.content_id,
                "edge_kind": "goal_parent",
            }
        )
        for dependency in subgoal.depends_on:
            goal_edges.append(
                {
                    "parent_goal_cid": goal_cids[dependency],
                    "child_goal_cid": subgoal.content_id,
                    "edge_kind": "goal_dependency",
                }
            )

    task_cids = {task.task_id: task.content_id for task in formal_plan.tasks}
    tasks: list[dict[str, Any]] = []
    for ordinal, task in enumerate(formal_plan.tasks, start=1):
        title, source_line, fields = block_map[task.task_id]
        dependencies = tuple(split_csv(fields.get("depends_on", "")))
        if dependencies != task.depends_on:
            raise MaterializationError(f"{task.task_id}: dependency projections differ")
        if fields.get("goal_id") != task.goal_id or fields.get("subgoal_id") != task.subgoal_id:
            raise MaterializationError(f"{task.task_id}: goal projection differs")
        if fields.get("board_namespace") != EXPECTED_NAMESPACE:
            raise MaterializationError(f"{task.task_id}: board namespace differs")
        construction_status = str(task.metadata.get("construction_status") or "")
        markdown_status = str(fields.get("status") or "")
        if construction_status.startswith("blocked_"):
            if markdown_status != "blocked" or construction_status not in fields.get(
                "blocked_reason", ""
            ):
                raise MaterializationError(f"{task.task_id}: blocked disposition differs")
            durable_status = "blocked"
        elif markdown_status == construction_status and construction_status in {
            "completed",
            "todo",
        }:
            durable_status = construction_status
        else:
            raise MaterializationError(f"{task.task_id}: construction status differs")
        schedulable = _metadata_bool(fields, "is_schedulable")
        review_only = _metadata_bool(fields, "review_only")
        if durable_status == "todo" and (not schedulable or review_only):
            raise MaterializationError(f"{task.task_id}: runnable task policy differs")
        if durable_status != "todo" and schedulable:
            raise MaterializationError(f"{task.task_id}: non-runnable task is schedulable")
        if construction_status.startswith("blocked_") and not review_only:
            raise MaterializationError(f"{task.task_id}: protected blocker is not review-only")
        outputs = tuple(split_csv(fields.get("outputs", "")))
        if outputs != tuple(split_csv(fields.get("predicted_files", ""))):
            raise MaterializationError(f"{task.task_id}: outputs and predicted files differ")
        markdown_metadata = dict(sorted(fields.items()))
        tasks.append(
            {
                "task_cid": task.content_id,
                "task_id": task.task_id,
                "task_alias": task.task_id,
                "goal_cid": goal_cids[task.subgoal_id],
                "plan_cid": formal_plan.content_id,
                "objective_id": "objective:lgcvf-root",
                "ordinal": ordinal,
                "status": durable_status,
                "priority": fields.get("priority", "P0"),
                "title": title,
                "dependencies": [task_cids[item] for item in task.depends_on],
                "outputs": [
                    {
                        "path": path,
                        "effect_id": content_identity(
                            {"formal_task_content_id": task.content_id, "path": path}
                        ),
                    }
                    for path in outputs
                ],
                "acceptance": [fields.get("acceptance", "")],
                "validations": [fields.get("validation", "")],
                "completion": fields.get("completion", "auto"),
                "review_only": review_only,
                "is_schedulable": schedulable,
                "blocked_reason": fields.get("blocked_reason", ""),
                "construction_status": construction_status,
                "formal_task_content_id": task.content_id,
                "formal_record": task.to_dict(),
                "markdown_metadata": markdown_metadata,
                "markdown_metadata_cid": content_identity(markdown_metadata),
                "source_line": source_line,
                "owning_repository": fields.get("owning_repository", ""),
                "board_namespace": EXPECTED_NAMESPACE,
            }
        )

    projection = config.get("initial_projection")
    if not isinstance(projection, Mapping):
        raise MaterializationError("initial_projection is required")
    if len(tasks) != projection.get("task_count") or len(goal_records) != projection.get(
        "goal_count"
    ):
        raise MaterializationError("population count differs from initial_projection")
    observed_completed = [item["task_id"] for item in tasks if item["status"] == "completed"]
    observed_blocked = [item["task_id"] for item in tasks if item["status"] == "blocked"]
    if observed_completed != projection.get("completed_task_ids"):
        raise MaterializationError("completed task projection differs")
    if observed_blocked != projection.get("blocked_task_ids"):
        raise MaterializationError("blocked task projection differs")

    population = {
        "schema": POPULATION_SCHEMA,
        "repository_tree_id": "git-tree:" + str(source["accelerator_tree"]),
        "source_head": str(source["accelerator_head"]),
        "source_forest_root": str(source["source_forest_root"]),
        "formal_repository_tree_id": formal_plan.repository_tree_id,
        "plan_root_cid": formal_plan.content_id,
        "objectives": goal_records,
        "goal_edges": goal_edges,
        "plans": [
            {
                "plan_cid": formal_plan.content_id,
                "plan_alias": EXPECTED_NAMESPACE,
                "goal_cid": root_goal.content_id,
                "status": "active",
                "repository_tree_id": "git-tree:" + str(source["accelerator_tree"]),
                "formal_repository_tree_id": formal_plan.repository_tree_id,
                "predecessor_plan_cid": formal_plan.metadata["predecessor_plan_cid"],
                "source_head": str(source["accelerator_head"]),
            }
        ],
        "tasks": tasks,
        "goal_cids_by_alias": goal_cids,
        "task_cids_by_alias": task_cids,
    }
    population["population_root"] = content_identity(population)
    return population


def build_population(
    config: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Load and bind the exact local formal and Markdown projections."""

    source = verify_source_binding(config, root=root)
    formal_value = config.get("formal_plan_path")
    todo_value = config.get("taskboard_path")
    try:
        _formal_path, formal_bytes, _formal_digest = _read_regular_evidence_bytes(
            root,
            formal_value,
            field="formal_plan_path",
        )
        _todo_path, todo_bytes, _todo_digest = _read_regular_evidence_bytes(
            root,
            todo_value,
            field="taskboard_path",
        )
        _require_head_bound_recovery_bytes(
            root,
            formal_value,
            formal_bytes,
            field="formal_plan_path",
        )
        _require_head_bound_recovery_bytes(
            root,
            todo_value,
            todo_bytes,
            field="taskboard_path",
        )
        formal_payload = _strict_json_loads(
            formal_bytes, noun="LGCVF formal plan"
        )
        formal_plan = FormalWorkPlan.from_dict(formal_payload)
        todo_text = todo_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError, MaterializationError, TypeError, ValueError) as exc:
        raise MaterializationError(f"LGCVF plan projection is unreadable: {exc}") from exc
    return project_population(
        config,
        formal_plan=formal_plan,
        todo_text=todo_text,
        source=source,
    )


def _paths(config: Mapping[str, Any], *, root: Path) -> dict[str, Path]:
    program = config.get("database_program")
    if not isinstance(program, Mapping):
        raise MaterializationError("database_program is required")
    control = _safe_path(root, program.get("store_id"), field="database_program.store_id")
    runtime = config.get("runtime_paths")
    if not isinstance(runtime, Mapping):
        raise MaterializationError("runtime_paths is required")
    evidence = _safe_path(root, runtime.get("evidence"), field="runtime_paths.evidence")
    return {
        "control": control,
        "coordination": control.with_name(f"{control.stem}.coordination.duckdb"),
        "execution": control.with_name(f"{control.stem}.execution.duckdb"),
        "receipt": evidence / "bootstrap" / "materialization.json",
    }


def _successor_paths(
    config: Mapping[str, Any], *, root: Path
) -> dict[str, Path]:
    """Resolve revision-store paths without changing the bootstrap contract."""

    paths = _paths(config, root=root)
    runtime = config.get("runtime_paths")
    if not isinstance(runtime, Mapping):
        raise MaterializationError("runtime_paths is required")
    state = _safe_path(root, runtime.get("state"), field="runtime_paths.state")
    evidence = _safe_path(
        root, runtime.get("evidence"), field="runtime_paths.evidence"
    )
    binding = config.get("plan_binding")
    if not isinstance(binding, Mapping):
        raise MaterializationError("plan_binding is required")
    predecessor = str(binding.get("predecessor_plan_cid") or "")
    if not predecessor:
        raise MaterializationError("plan_binding.predecessor_plan_cid is required")
    formal_path = _safe_path(
        root, config.get("formal_plan_path"), field="formal_plan_path"
    )
    paths.update(
        {
            "revision_store": state / "plan-revision-store",
            "revision_receipts": evidence / "plan-revisions",
            "predecessor_archive": (
                formal_path.parent / "plan_revisions" / f"{predecessor}.json"
            ),
        }
    )
    return paths


def _plain_json(value: Any) -> Any:
    """Return a detached canonical JSON value."""

    def thaw(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): thaw(child) for key, child in item.items()}
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            return [thaw(child) for child in item]
        return item

    return json.loads(_canonical_bytes(thaw(value)))


def _load_predecessor_plan_evidence(
    config: Mapping[str, Any], *, root: Path
) -> tuple[FormalWorkPlan, str]:
    """Read and bind the immutable predecessor archive from one held FD."""

    binding = config.get("plan_binding")
    if not isinstance(binding, Mapping):
        raise MaterializationError("plan_binding is required")
    predecessor_cid = str(binding.get("predecessor_plan_cid") or "")
    if int(binding.get("plan_revision") or 0) != 2:
        raise MaterializationError("successor continuation requires plan_revision 2")
    formal_path = Path(str(config.get("formal_plan_path") or ""))
    if (
        not formal_path.parts
        or formal_path.is_absolute()
        or ".." in formal_path.parts
    ):
        raise MaterializationError("formal_plan_path is unsafe")
    archive_relative = (
        formal_path.parent / "plan_revisions" / f"{predecessor_cid}.json"
    )
    _, archive_bytes, archive_sha256 = _read_regular_evidence_bytes(
        root,
        archive_relative.as_posix(),
        field="immutable predecessor archive",
    )
    try:
        payload = _strict_json_loads(
            archive_bytes, noun="immutable predecessor archive"
        )
        predecessor = FormalWorkPlan.from_dict(payload)
    except (MaterializationError, TypeError, ValueError) as exc:
        raise MaterializationError("immutable predecessor archive is unreadable") from exc
    if predecessor.content_id != predecessor_cid:
        raise MaterializationError("immutable predecessor archive identity differs")
    if predecessor.metadata.get("board_namespace") != EXPECTED_NAMESPACE:
        raise MaterializationError("predecessor archive namespace differs")
    return predecessor, archive_sha256


def _load_predecessor_plan(
    config: Mapping[str, Any], *, root: Path
) -> FormalWorkPlan:
    return _load_predecessor_plan_evidence(config, root=root)[0]


def _task_map(
    projection: Mapping[str, Any], *, noun: str
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    raw = projection.get("tasks")
    if not isinstance(raw, list):
        raise MaterializationError(f"{noun} has no typed task population")
    by_cid: dict[str, Mapping[str, Any]] = {}
    by_alias: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise MaterializationError(f"{noun} contains a malformed task")
        cid = str(item.get("task_cid") or "")
        alias = str(item.get("task_alias") or item.get("task_id") or "")
        if not cid or not alias or cid in by_cid or alias in by_alias:
            raise MaterializationError(f"{noun} task identity is missing or duplicated")
        by_cid[cid] = item
        by_alias[alias] = item
    return by_cid, by_alias


def _task_lifecycle(status: Any) -> LifecycleState:
    normalized = str(status or "").strip().lower()
    if normalized in {"todo", "pending", "queued", "retrying"}:
        return LifecycleState.UNSTARTED
    if normalized == "ready":
        return LifecycleState.READY
    if normalized == "blocked":
        return LifecycleState.BLOCKED
    if normalized == "claimed":
        return LifecycleState.CLAIMED
    if normalized in {"in_progress", "running"}:
        return LifecycleState.RUNNING
    if normalized in {"completed", "complete", "done", "skipped"}:
        return LifecycleState.COMPLETED
    if normalized in {"failed", "cancelled", "quarantined", "rejected"}:
        return LifecycleState.FAILED
    raise MaterializationError(f"task status {normalized!r} has no revision lifecycle")


def _assert_receipt_bound_task_spec(
    task: Mapping[str, Any],
    *,
    expected_legacy_spec_cid: str,
    revision_history: Mapping[str, Any],
) -> None:
    """Admit lifecycle receipt evolution without weakening plan-spec CAS.

    Revision-2 receipts predate the authority-only task projection and bind
    the legacy spec CID, which includes ``body.completion_receipt``.  Locate
    that exact receipt-bound body in the append-only task revision history,
    then compare its authority projection with the current task.  Thus an
    admitted status receipt may advance while title, authority, identity,
    relations, validation, and acceptance fields remain immutable.
    """

    task_cid = str(task.get("task_cid") or "")
    if revision_history.get("task_cid") != task_cid:
        raise MaterializationError(f"{task_cid}: task revision history differs")
    revisions = revision_history.get("revisions")
    if not isinstance(revisions, list) or not revisions:
        raise MaterializationError(f"{task_cid}: task revision history is absent")
    current_revision = int(task.get("revision") or 0)
    current_status = str(task.get("status") or "")
    current_body = _plain_json(task.get("body") or {})
    current_rows = [
        row
        for row in revisions
        if isinstance(row, Mapping)
        and int(row.get("revision") or 0) == current_revision
        and str(row.get("status") or "") == current_status
        and _plain_json(row.get("body") or {}) == current_body
    ]
    if len(current_rows) != 1:
        raise MaterializationError(
            f"{task_cid}: current lifecycle state is not revision-bound"
        )
    if str(task.get("spec_cid") or "") == expected_legacy_spec_cid:
        return

    baseline_authority_cids: set[str] = set()
    for row in revisions:
        if not isinstance(row, Mapping):
            raise MaterializationError(f"{task_cid}: task revision is malformed")
        candidate = _plain_json(task)
        candidate["body"] = _plain_json(row.get("body") or {})
        if task_projection_spec_cid(candidate) == expected_legacy_spec_cid:
            baseline_authority_cids.add(task_authority_spec_cid(candidate))
    if not baseline_authority_cids:
        raise MaterializationError(f"{task_cid}: receipt-bound task spec is absent")
    if baseline_authority_cids != {task_authority_spec_cid(task)}:
        raise MaterializationError(f"{task_cid}: authority-bearing task spec drifted")


def _raw_task_from_projection(task: Mapping[str, Any]) -> dict[str, Any]:
    """Losslessly adapt an IntentRepository task back to materializer input."""

    body = task.get("body")
    identity = task.get("identity")
    if not isinstance(body, Mapping) or not isinstance(identity, Mapping):
        raise MaterializationError("live task lacks typed body or identity")
    forbidden_body = {
        "task_cid",
        "task_id",
        "task_alias",
        "goal_cid",
        "dependencies",
        "outputs",
        "acceptance",
        "validations",
        "status",
        "priority",
        "ordinal",
        "plan_cid",
        "objective_id",
    }
    if forbidden_body & set(body):
        raise MaterializationError("live task body collides with projection fields")
    dependencies: list[str] = []
    for dependency in task.get("dependencies") or ():
        if not isinstance(dependency, Mapping):
            raise MaterializationError("live task dependency is malformed")
        if str(dependency.get("kind") or "depends_on") != "depends_on":
            raise MaterializationError("non-default dependency kind is not losslessly adaptable")
        dependency_cid = str(dependency.get("dependency_task_cid") or "")
        if not dependency_cid:
            raise MaterializationError("live task dependency identity is empty")
        dependencies.append(dependency_cid)
    outputs: list[dict[str, Any]] = []
    for output in task.get("outputs") or ():
        if not isinstance(output, Mapping) or not isinstance(output.get("effect"), Mapping):
            raise MaterializationError("live task output is malformed")
        effect = _plain_json(output["effect"])
        if effect.get("path") != output.get("path"):
            raise MaterializationError("live output effect/path projection differs")
        outputs.append(effect)
    acceptance: list[dict[str, Any]] = []
    for entry in task.get("acceptance") or ():
        if not isinstance(entry, Mapping) or not isinstance(
            entry.get("evidence_policy"), Mapping
        ):
            raise MaterializationError("live task acceptance is malformed")
        policy = _plain_json(entry["evidence_policy"])
        if policy.get("criterion") != entry.get("criterion"):
            raise MaterializationError("acceptance policy cannot be losslessly adapted")
        acceptance.append(policy)
    validations: list[dict[str, Any]] = []
    for entry in task.get("validations") or ():
        if not isinstance(entry, Mapping) or not isinstance(entry.get("policy"), Mapping):
            raise MaterializationError("live task validation is malformed")
        argv = entry.get("argv")
        if not isinstance(argv, list) or not all(isinstance(part, str) for part in argv):
            raise MaterializationError("live validation argv is malformed")
        validations.append({"argv": list(argv), **_plain_json(entry["policy"])})
    expected_identity = {
        "task_cid": str(task.get("task_cid") or ""),
        "task_alias": str(task.get("task_alias") or ""),
        "repository_tree_id": str(identity.get("repository_tree_id") or ""),
    }
    if dict(identity) != expected_identity or not expected_identity["repository_tree_id"]:
        raise MaterializationError("live task identity is not the canonical LGCVF shape")
    return {
        "task_cid": expected_identity["task_cid"],
        "task_id": expected_identity["task_alias"],
        "task_alias": expected_identity["task_alias"],
        "goal_cid": str(task.get("goal_cid") or ""),
        "plan_cid": str(task.get("plan_cid") or ""),
        "objective_id": str(task.get("objective_id") or ""),
        "ordinal": int(task.get("ordinal") or 0),
        "status": str(task.get("status") or ""),
        "priority": str(task.get("priority") or ""),
        "dependencies": dependencies,
        "outputs": outputs,
        "acceptance": acceptance,
        "validations": validations,
        **_plain_json(body),
    }


def _read_successor_state(
    config: Mapping[str, Any], *, root: Path
) -> dict[str, Any]:
    """Read all continuation evidence and prove that the read changed no store."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    paths = _successor_paths(config, root=root)
    stores = {key: paths[key] for key in ("control", "coordination", "execution")}
    missing = [key for key, path in stores.items() if not path.is_file()]
    if missing:
        raise MaterializationError(f"operational stores are absent: {', '.join(missing)}")
    before = {
        key: (path.stat().st_size, path.stat().st_mtime_ns, _sha256_file(path))
        for key, path in stores.items()
    }
    source = DatabaseTaskSource(
        paths["control"],
        owner_id="lgcvf-successor:read-only-observer",
        install_schema=False,
    )
    try:
        plan_projection = _plain_json(source.plan_projection())
        completion_projection = _plain_json(source.completion_evidence_projection())
        revision_histories = {
            str(task.get("task_cid") or ""): _plain_json(
                source.task_revision_history_projection(
                    str(task.get("task_cid") or "")
                )
            )
            for task in plan_projection.get("tasks") or ()
            if isinstance(task, Mapping)
        }
    finally:
        source.close()
    try:
        coordination = read_coordination_registry_projection(paths["coordination"])
    except Exception as exc:
        raise MaterializationError("coordination registry fails typed verification") from exc
    execution = _read_only_execution(paths["execution"], expected_stage="live")
    bootstrap = _load_receipt(paths["receipt"])
    after = {
        key: (path.stat().st_size, path.stat().st_mtime_ns, _sha256_file(path))
        for key, path in stores.items()
    }
    if before != after:
        raise MaterializationError("successor observation changed an operational store")
    composite = {
        "schema": SUCCESSOR_COMPOSITE_PROJECTION_SCHEMA,
        "plan_projection_cid": str(plan_projection.get("projection_cid") or ""),
        "completion_projection_cid": str(
            completion_projection.get("projection_cid") or ""
        ),
        "coordination_projection_root": str(coordination.get("projection_root") or ""),
    }
    composite["projection_cid"] = content_identity(composite)
    return {
        "plan_projection": plan_projection,
        "task_revision_histories": revision_histories,
        "completion_projection": completion_projection,
        "coordination_projection": coordination,
        "execution_projection": execution,
        "bootstrap_receipt": bootstrap,
        "store_observations": {
            key: {"size": value[0], "mtime_ns": value[1], "sha256": value[2]}
            for key, value in before.items()
        },
        "composite_projection": composite,
    }


def _retained_completion_binding(
    state: Mapping[str, Any], completed_cids: Sequence[str]
) -> dict[str, Any]:
    completed = set(completed_cids)
    completion = state.get("completion_projection")
    coordination = state.get("coordination_projection")
    if not isinstance(completion, Mapping) or not isinstance(coordination, Mapping):
        raise MaterializationError("completion binding projections are malformed")
    binding = {
        "task_states": sorted(
            [
                _plain_json(item)
                for item in completion.get("task_states") or ()
                if isinstance(item, Mapping) and item.get("task_cid") in completed
            ],
            key=lambda item: str(item.get("task_cid") or ""),
        ),
        "completion_receipts": sorted(
            [
                _plain_json(item)
                for item in completion.get("completion_receipts") or ()
                if isinstance(item, Mapping) and item.get("task_cid") in completed
            ],
            key=lambda item: (
                str(item.get("task_cid") or ""),
                str(item.get("receipt_cid") or ""),
            ),
        ),
        "logical_completions": sorted(
            [
                _plain_json(item)
                for item in coordination.get("logical_completions") or ()
                if isinstance(item, Mapping) and item.get("task_cid") in completed
            ],
            key=lambda item: str(item.get("task_cid") or ""),
        ),
    }
    if {str(item.get("task_cid") or "") for item in binding["task_states"]} != completed:
        raise MaterializationError("retained completion states are incomplete")
    if {
        str(item.get("task_cid") or "") for item in binding["logical_completions"]
    } != completed:
        raise MaterializationError("retained logical completions are incomplete")
    binding["binding_cid"] = content_identity(binding)
    return binding


def _protected_blocker_binding(
    task_by_alias: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for alias in sorted(SUCCESSOR_REPRIORITIZED_ALIASES):
        task = task_by_alias.get(alias)
        if not isinstance(task, Mapping):
            raise MaterializationError(f"{alias}: protected blocker is absent")
        tasks[alias] = {
            key: _plain_json(value)
            for key, value in task.items()
            if key not in {"ordinal", "revision", "plan_cid", "spec_cid"}
        }
    binding = {"tasks": tasks}
    binding["binding_cid"] = content_identity(binding)
    return binding


def _assert_quiescent_predecessor(
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    predecessor: FormalWorkPlan,
    root: Path,
) -> dict[str, Any]:
    """Validate the exact revision-1 population and all operational fences."""

    plan = state["plan_projection"]
    completion = state["completion_projection"]
    coordination = state["coordination_projection"]
    if not all(isinstance(item, Mapping) for item in (plan, completion, coordination)):
        raise MaterializationError("successor state projections are malformed")
    binding = config.get("plan_binding")
    if not isinstance(binding, Mapping):
        raise MaterializationError("plan_binding is required")
    predecessor_cid = str(binding.get("predecessor_plan_cid") or "")
    active_plans = [
        item
        for item in plan.get("plans") or ()
        if isinstance(item, Mapping) and item.get("status") == "active"
    ]
    if len(active_plans) != 1 or active_plans[0].get("plan_cid") != predecessor_cid:
        raise MaterializationError("control store is not at the exact active predecessor")
    if state["bootstrap_receipt"].get("plan_root_cid") != predecessor_cid:
        raise MaterializationError("bootstrap receipt does not bind the predecessor plan")

    live_by_cid, live_by_alias = _task_map(plan, noun="live predecessor projection")
    expected_by_alias = {task.task_id: task.content_id for task in predecessor.tasks}
    observed_by_alias = {
        alias: str(task.get("task_cid") or "") for alias, task in live_by_alias.items()
    }
    if observed_by_alias != expected_by_alias:
        raise MaterializationError("live predecessor logical task population differs")
    if SUCCESSOR_ADDED_ALIAS in live_by_alias:
        raise MaterializationError("successor task already exists under the predecessor")

    counts = coordination.get("counts")
    if not isinstance(counts, Mapping):
        raise MaterializationError("coordination counts are absent")
    active_fields = (
        "active_task_claims",
        "active_task_attempts",
        "active_fenced_leases",
        "active_resource_claims",
        "active_maintenance_leases",
    )
    active = {field: int(counts.get(field) or 0) for field in active_fields}
    if any(active.values()):
        raise MaterializationError(
            "active claims, attempts, leases, or writer reservations block successor apply"
        )
    prepared = [
        item
        for item in coordination.get("logical_completions") or ()
        if isinstance(item, Mapping) and item.get("status") != "succeeded"
    ]
    if prepared:
        raise MaterializationError("prepared or non-success completion blocks successor apply")

    registered = {
        str(item.get("task_cid") or ""): str(item.get("task_id") or "")
        for item in coordination.get("tasks") or ()
        if isinstance(item, Mapping)
    }
    expected_registered = {
        cid: alias for alias, cid in expected_by_alias.items()
    }
    if registered != expected_registered:
        raise MaterializationError("coordination predecessor task registry differs")
    expected_edges = {
        (cid, str(dependency.get("dependency_task_cid") or ""))
        for cid, task in live_by_cid.items()
        for dependency in task.get("dependencies") or ()
        if isinstance(dependency, Mapping)
    }
    observed_edges = {
        (str(item.get("task_cid") or ""), str(item.get("dependency_task_cid") or ""))
        for item in coordination.get("dependency_edges") or ()
        if isinstance(item, Mapping)
    }
    if observed_edges != expected_edges:
        raise MaterializationError("coordination predecessor dependency graph differs")

    task_states = {
        str(item.get("task_cid") or ""): item
        for item in completion.get("task_states") or ()
        if isinstance(item, Mapping)
    }
    if set(task_states) != set(live_by_cid):
        raise MaterializationError("completion evidence task population differs")
    for cid, task in live_by_cid.items():
        if task_states[cid].get("status") != task.get("status"):
            raise MaterializationError("control and completion task states differ")
    completed_cids = {
        cid
        for cid, task in live_by_cid.items()
        if _task_lifecycle(task.get("status")) is LifecycleState.COMPLETED
    }
    coordination_completed = {
        str(item.get("task_cid") or "")
        for item in coordination.get("logical_completions") or ()
        if isinstance(item, Mapping) and item.get("status") == "succeeded"
    }
    if coordination_completed != completed_cids:
        raise MaterializationError("completion evidence disagrees across control stores")
    receipt_task_cids = {
        str(item.get("task_cid") or "")
        for item in completion.get("completion_receipts") or ()
        if isinstance(item, Mapping)
    }
    live_alias_by_cid = {
        cid: str(task.get("task_alias") or "") for cid, task in live_by_cid.items()
    }
    live_revision_by_cid = {
        cid: int(task.get("revision") or 0) for cid, task in live_by_cid.items()
    }
    for logical in coordination.get("logical_completions") or ():
        if not isinstance(logical, Mapping):
            raise MaterializationError("logical completion record is malformed")
        cid = str(logical.get("task_cid") or "")
        body = logical.get("body")
        if cid not in receipt_task_cids and (
            not isinstance(body, Mapping)
            or body
            != {
                "authority": "database_population",
                "source_status": "completed",
                "task_alias": live_alias_by_cid.get(cid),
                "task_revision": live_revision_by_cid.get(cid),
            }
        ):
            raise MaterializationError("bootstrap completion evidence is stale or rewritten")
    for receipt in completion.get("completion_receipts") or ():
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("task_cid") not in completed_cids
            or not receipt.get("receipt_cid")
            or not receipt.get("evidence_digest")
        ):
            raise MaterializationError("runtime completion receipt is stale or malformed")

    for alias, authority, completion_mode in (
        ("LGCVF-121", "blocked_external_authority", "external-authority"),
        ("LGCVF-123", "blocked_manual", "manual"),
    ):
        task = live_by_alias.get(alias)
        body = task.get("body") if isinstance(task, Mapping) else None
        if (
            not isinstance(task, Mapping)
            or task.get("status") != "blocked"
            or not isinstance(body, Mapping)
            or body.get("construction_status") != authority
            or body.get("completion") != completion_mode
            or body.get("review_only") is not True
        ):
            raise MaterializationError(f"{alias}: protected blocker semantics differ")
    tree_ids = {
        str(task.get("identity", {}).get("repository_tree_id") or "")
        for task in live_by_cid.values()
        if isinstance(task.get("identity"), Mapping)
    }
    if len(tree_ids) != 1 or not next(iter(tree_ids), ""):
        raise MaterializationError("predecessor task identities have inconsistent tree roots")
    evidence = {
        "predecessor_plan_cid": predecessor.content_id,
        "predecessor_archive_sha256": _sha256_file(
            _successor_paths(config, root=root)["predecessor_archive"]
        ),
        "plan_projection_cid": str(plan.get("projection_cid") or ""),
        "completion_projection_cid": str(completion.get("projection_cid") or ""),
        "coordination_projection_root": str(coordination.get("projection_root") or ""),
        "bootstrap_receipt_cid": str(
            state["bootstrap_receipt"].get("receipt_cid") or ""
        ),
        "completed_task_cids": sorted(completed_cids),
        "blocked_task_cids": sorted(
            cid
            for cid, task in live_by_cid.items()
            if _task_lifecycle(task.get("status")) is LifecycleState.BLOCKED
        ),
        "task_spec_cids": {
            alias: str(task.get("spec_cid") or "")
            for alias, task in sorted(live_by_alias.items())
        },
        "base_repository_tree_id": next(iter(tree_ids)),
        "active_counts": active,
        "retained_completion_binding": _retained_completion_binding(
            state, sorted(completed_cids)
        ),
        "protected_blocker_binding": _protected_blocker_binding(live_by_alias),
    }
    evidence["evidence_root"] = content_identity(evidence)
    return evidence


def _successor_candidate_population(
    population: Mapping[str, Any],
    live_plan: Mapping[str, Any],
    *,
    predecessor: FormalWorkPlan,
    base_repository_tree_id: str,
) -> dict[str, Any]:
    """Build the narrow operational revision without rewriting narrative history."""

    _live_by_cid, live_by_alias = _task_map(
        live_plan, noun="live predecessor projection"
    )
    desired_by_alias = {
        str(task.get("task_id") or ""): task for task in population.get("tasks") or ()
    }
    desired_alias_by_cid = {
        str(task.get("task_cid") or ""): alias
        for alias, task in desired_by_alias.items()
    }
    predecessor_aliases = {task.task_id for task in predecessor.tasks}
    expected_aliases = predecessor_aliases | {SUCCESSOR_ADDED_ALIAS}
    if set(desired_by_alias) != expected_aliases:
        raise MaterializationError("revision-2 desired task population differs")
    candidate_tasks: list[dict[str, Any]] = []
    for desired in population.get("tasks") or ():
        if not isinstance(desired, Mapping):
            raise MaterializationError("revision-2 task projection is malformed")
        alias = str(desired.get("task_id") or "")
        if alias == SUCCESSOR_ADDED_ALIAS:
            candidate = _plain_json(desired)
        elif alias in SUCCESSOR_AMENDED_ALIASES:
            live = live_by_alias[alias]
            lifecycle = _task_lifecycle(live.get("status"))
            if lifecycle not in {
                LifecycleState.UNSTARTED,
                LifecycleState.READY,
            }:
                raise MaterializationError(f"{alias}: started history cannot be amended")
            candidate = _plain_json(desired)
            candidate["status"] = str(live.get("status") or "")
        elif alias in SUCCESSOR_REPRIORITIZED_ALIASES:
            live = live_by_alias[alias]
            if _task_lifecycle(live.get("status")) is not LifecycleState.BLOCKED:
                raise MaterializationError(f"{alias}: protected blocker is not blocked")
            candidate = _raw_task_from_projection(live)
            candidate["ordinal"] = int(desired.get("ordinal") or 0)
        else:
            candidate = _raw_task_from_projection(live_by_alias[alias])
        if alias in live_by_alias:
            candidate["task_cid"] = str(live_by_alias[alias]["task_cid"])
            candidate["task_id"] = alias
            candidate["task_alias"] = alias
        translated_dependencies: list[str] = []
        for dependency_cid in candidate.get("dependencies") or ():
            dependency_alias = desired_alias_by_cid.get(str(dependency_cid))
            if dependency_alias is None:
                translated_dependencies.append(str(dependency_cid))
            elif dependency_alias in live_by_alias:
                translated_dependencies.append(
                    str(live_by_alias[dependency_alias]["task_cid"])
                )
            else:
                translated_dependencies.append(str(dependency_cid))
        for dependency_alias in SUCCESSOR_RUNTIME_DEPENDENCIES.get(alias, ()):
            dependency = desired_by_alias.get(dependency_alias)
            if not isinstance(dependency, Mapping):
                raise MaterializationError(
                    f"{alias}: runtime dependency {dependency_alias} is absent"
                )
            dependency_cid = str(dependency.get("task_cid") or "")
            if not dependency_cid:
                raise MaterializationError(
                    f"{alias}: runtime dependency {dependency_alias} has no identity"
                )
            if dependency_cid not in translated_dependencies:
                translated_dependencies.append(dependency_cid)
        candidate["dependencies"] = translated_dependencies
        candidate_tasks.append(candidate)
    candidate_population = _plain_json(population)
    candidate_population["repository_tree_id"] = base_repository_tree_id
    candidate_population["tasks"] = candidate_tasks
    candidate_population["task_cids_by_alias"] = {
        str(task["task_id"]): str(task["task_cid"]) for task in candidate_tasks
    }
    candidate_population.pop("population_root", None)
    candidate_population["population_root"] = content_identity(candidate_population)
    return candidate_population


def _project_candidate(
    candidate_population: Mapping[str, Any],
    *,
    base_repository_tree_id: str,
) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    with tempfile.TemporaryDirectory(prefix="lgcvf-successor-preview-") as temporary:
        source = DatabaseTaskSource(Path(temporary) / "candidate.duckdb")
        try:
            source.materialize(
                candidate_population,
                repository_tree_id=base_repository_tree_id,
                plan_root_cid=str(candidate_population.get("plan_root_cid") or ""),
            )
            return _plain_json(source.plan_projection())
        finally:
            source.close()


def _population_digest(
    kind: PopulationKind, members: Sequence[str] = ()
) -> PlanPopulationDigest:
    return PlanPopulationDigest(kind=kind, member_cids=tuple(members))


def _revision_roots(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    program_root: str,
) -> PlanAuthorityRoots:
    provider = config.get("provider")
    if not isinstance(provider, Mapping):
        provider = {}
    return PlanAuthorityRoots(
        repository_id="repository:ipfs-accelerate-py",
        repository_root_cid=str(population.get("repository_tree_id") or ""),
        dirty_worktree_root=str(population.get("source_forest_root") or ""),
        task_source_id=content_identity(
            {
                "interface": "DatabaseTaskSource@1",
                "board_namespace": EXPECTED_NAMESPACE,
            }
        ),
        task_source_revision=str(
            state["composite_projection"].get("projection_cid") or ""
        ),
        policy_root=content_identity(
            {
                "protected_paths": list(config.get("protected_paths") or ()),
                "writer_policy": config.get("bootstrap_writer_policy"),
            }
        ),
        intent_ir_root=str(state["plan_projection"].get("projection_cid") or ""),
        legal_ir_root=content_identity(
            {"authority": "datasets", "status": "not_extended_by_successor"}
        ),
        security_ir_root=content_identity(
            {"authority": "datasets", "status": "not_extended_by_successor"}
        ),
        program_root=program_root,
        capability_catalog_root=content_identity(
            {
                "schema_revision": EXPECTED_SCHEMA_REVISION,
                "schema_profile": EXPECTED_SCHEMA_PROFILE,
                "quack_qualified": False,
            }
        ),
        provider_catalog_root=content_identity(
            {
                "provider": provider.get("name"),
                "route": provider.get("route"),
            }
        ),
        usage_policy_root=content_identity(
            {
                "maximum_writer_processes": 1,
                "automatic_installation_permitted": False,
                "production_authorized": False,
            }
        ),
        configuration_root=content_identity(_plain_json(config)),
    )


def _revision_contract(
    *,
    plan_root_cid: str,
    semantic_revision: int,
    parent_plan_root: str,
    origin: PlanOrigin,
    roots: PlanAuthorityRoots,
    request_cid: str,
    delta_cid: str,
    evidence_root: str,
    admission_receipt_cid: str,
    goal_cids: Sequence[str],
    task_cids: Sequence[str],
    added_cids: Sequence[str],
    retained_cids: Sequence[str],
    claimed_cids: Sequence[str],
    completed_cids: Sequence[str],
    blocked_cids: Sequence[str],
    control_path: str,
    coordination_path: str,
) -> PlanRevision:
    return PlanRevision(
        plan_root_cid=plan_root_cid,
        semantic_revision=semantic_revision,
        parent_plan_root=parent_plan_root,
        origin=origin,
        roots=roots,
        request_cid=request_cid,
        delta_cid=delta_cid,
        scan_receipt_cid=evidence_root,
        query_plan_cid="",
        evidence_bundle_cid=evidence_root,
        admission_receipt_cid=admission_receipt_cid,
        execution_plan_cid="",
        goal_population=_population_digest(PopulationKind.RETAINED, goal_cids),
        task_population=_population_digest(PopulationKind.RETAINED, task_cids),
        added_population=_population_digest(PopulationKind.ADDED, added_cids),
        superseded_population=_population_digest(PopulationKind.SUPERSEDED),
        retained_population=_population_digest(PopulationKind.RETAINED, retained_cids),
        deferred_population=_population_digest(PopulationKind.DEFERRED),
        claimed_population=_population_digest(PopulationKind.CLAIMED, claimed_cids),
        completed_population=_population_digest(
            PopulationKind.COMPLETED, completed_cids
        ),
        blocked_population=_population_digest(PopulationKind.BLOCKED, blocked_cids),
        resource_contract=PlanResourceContract(
            resource_class="cpu-small",
            resource_stage="plan-steer",
            cpu_slots=1,
            process_slots=1,
        ),
        provider_contract=PlanProviderContract(
            provider_requirement="",
            endpoint_policy_class="none",
        ),
        lease_contract=PlanLeaseContract(
            lease_scope="task-source",
            owner_identity_rule="single-writer-materializer",
        ),
        retry_contract=PlanRetryContract(
            max_retries=0,
            compensation_policy="exact-byte-restore",
        ),
        worktree_contract=PlanWorktreeContract(
            policy="require-clean",
            isolation_required=False,
        ),
        merge_strategy=PlanMergeStrategy(kind=MergeStrategyKind.SERIAL),
        conflict_contract=PlanConflictContract(
            predicted_files=(control_path, coordination_path),
            exclusive_paths=(control_path, coordination_path),
            max_files=2,
        ),
        completion_rule=PlanCompletionRule(
            authority=CompletionAuthority.VALIDATION_GATE,
            required_evidence_kinds=("typed-store-projection", "exact-byte-rollback"),
        ),
        validation_dag=(),
        rollback_ref=evidence_root,
        event_cursor=evidence_root,
    )


def preview_successor(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Construct a body-free, read-only revision-2 continuation proposal."""

    if _targets_fresh_recovery_generation(config, root=root):
        raise MaterializationError(
            "fresh recovery targets reject legacy successor preview"
        )

    predecessor = _load_predecessor_plan(config, root=root)
    state = _read_successor_state(config, root=root)
    evidence = _assert_quiescent_predecessor(
        config,
        state,
        predecessor=predecessor,
        root=root,
    )
    candidate_population = _successor_candidate_population(
        population,
        state["plan_projection"],
        predecessor=predecessor,
        base_repository_tree_id=str(evidence["base_repository_tree_id"]),
    )
    candidate_projection = _project_candidate(
        candidate_population,
        base_repository_tree_id=str(evidence["base_repository_tree_id"]),
    )
    live_by_cid, live_by_alias = _task_map(
        state["plan_projection"], noun="live predecessor projection"
    )
    candidate_by_cid, candidate_by_alias = _task_map(
        candidate_projection, noun="candidate successor projection"
    )
    added_cids = set(candidate_by_cid) - set(live_by_cid)
    expected_added_cid = str(candidate_by_alias[SUCCESSOR_ADDED_ALIAS]["task_cid"])
    if added_cids != {expected_added_cid}:
        raise MaterializationError("successor adds a task other than LGCVF-113")
    changed_aliases = {
        alias
        for alias in live_by_alias
        if live_by_alias[alias].get("spec_cid")
        != candidate_by_alias[alias].get("spec_cid")
    }
    if changed_aliases != SUCCESSOR_CHANGED_ALIASES:
        raise MaterializationError(
            "successor changed task specifications outside its closed amendment set: "
            + ", ".join(sorted(changed_aliases ^ SUCCESSOR_CHANGED_ALIASES))
        )
    if _protected_blocker_binding(live_by_alias)["binding_cid"] != (
        _protected_blocker_binding(candidate_by_alias)["binding_cid"]
    ):
        raise MaterializationError(
            "protected blocker amendment changed more than its ordinal"
        )
    for alias in set(live_by_alias) - SUCCESSOR_CHANGED_ALIASES:
        if live_by_alias[alias].get("task_cid") != candidate_by_alias[alias].get(
            "task_cid"
        ):
            raise MaterializationError(f"{alias}: retained logical identity changed")

    paths = _successor_paths(config, root=root)
    control_relative = paths["control"].relative_to(root).as_posix()
    coordination_relative = paths["coordination"].relative_to(root).as_posix()
    roots = _revision_roots(
        config,
        population,
        state,
        program_root=str(population["plan_root_cid"]),
    )
    request_body = {
        "operation": "continue-lgcvf-revision-2",
        "base_plan_root": predecessor.content_id,
        "candidate_plan_root": str(population["plan_root_cid"]),
        "evidence_root": str(evidence["evidence_root"]),
    }
    request_cid = content_identity(request_body)
    completed_cids = tuple(str(item) for item in evidence["completed_task_cids"])
    blocked_cids = tuple(str(item) for item in evidence["blocked_task_cids"])
    claimed_population = _population_digest(PopulationKind.CLAIMED)
    completed_population = _population_digest(
        PopulationKind.COMPLETED, completed_cids
    )
    items: list[PlanDeltaItem] = []
    for alias in sorted(SUCCESSOR_CHANGED_ALIASES):
        live = live_by_alias[alias]
        candidate = candidate_by_alias[alias]
        operation = (
            PlanDeltaOperation.REPRIORITIZE_UNSTARTED_TASK
            if alias in SUCCESSOR_REPRIORITIZED_ALIASES
            else PlanDeltaOperation.AMEND_UNSTARTED_TASK
        )
        effect = f"{operation.value}:{alias}"
        items.append(
            PlanDeltaItem(
                item_key=f"revision-2-{alias.lower()}",
                operation=operation,
                target_cid=str(live["task_cid"]),
                expected_target_lifecycle=_task_lifecycle(live.get("status")),
                expected_target_spec_revision=str(live.get("spec_cid") or ""),
                before_digest=str(live.get("spec_cid") or ""),
                after_record_cid=str(candidate.get("spec_cid") or ""),
                effect_class=DeltaEffectClass.MATERIALIZABLE_NOW,
                rationale=(
                    "Shift the protected blocker ordinal without changing its authority."
                    if operation is PlanDeltaOperation.REPRIORITIZE_UNSTARTED_TASK
                    else (
                        "Apply reviewed metadata and gate launch on LGCVF-113."
                        if alias == "LGCVF-120"
                        else "Apply the reviewed revision-2 operational metadata amendment."
                    )
                ),
                provenance={
                    "predecessor_plan_cid": predecessor.content_id,
                    "candidate_plan_cid": population["plan_root_cid"],
                    "evidence_root": evidence["evidence_root"],
                },
                expected_effects=(effect,),
                rollback_refs=(str(evidence["evidence_root"]),),
                affected_task_cids=(str(live["task_cid"]),),
            )
        )
    add_effect = f"add_task:{SUCCESSOR_ADDED_ALIAS}"
    items.append(
        PlanDeltaItem(
            item_key="revision-2-add-lgcvf-113",
            operation=PlanDeltaOperation.ADD_TASK,
            target_cid="",
            expected_target_lifecycle=LifecycleState.PROPOSED,
            expected_target_spec_revision="",
            before_digest="",
            after_record_cid=expected_added_cid,
            effect_class=DeltaEffectClass.MATERIALIZABLE_NOW,
            rationale="Add the independently judged hermetic qualification task.",
            provenance={
                "predecessor_plan_cid": predecessor.content_id,
                "candidate_plan_cid": population["plan_root_cid"],
                "evidence_root": evidence["evidence_root"],
            },
            expected_effects=(add_effect,),
            rollback_refs=(str(evidence["evidence_root"]),),
            affected_task_cids=(expected_added_cid,),
        )
    )
    expected_effects = tuple(
        effect for item in items for effect in item.expected_effects
    )
    admission_body = {
        "request_cid": request_cid,
        "predecessor_archive": predecessor.content_id,
        "evidence_root": evidence["evidence_root"],
        "changed_aliases": sorted(SUCCESSOR_CHANGED_ALIASES),
        "added_aliases": [SUCCESSOR_ADDED_ALIAS],
        "runtime_dependency_edges": [
            [task_alias, dependency_alias]
            for task_alias, dependency_aliases in sorted(
                SUCCESSOR_RUNTIME_DEPENDENCIES.items()
            )
            for dependency_alias in dependency_aliases
        ],
        "active_authority_count": 0,
        "history_rewritten": False,
    }
    admission_cid = content_identity(admission_body)
    delta = PlanDelta(
        base_plan_root=predecessor.content_id,
        base_plan_revision=1,
        request_cid=request_cid,
        roots=roots,
        items=tuple(items),
        expected_effects=expected_effects,
        deferred_item_keys=(),
        claimed_population_digest=claimed_population.digest,
        accepted_population_digest=completed_population.digest,
        scan_receipt_cid=str(evidence["evidence_root"]),
        evidence_bundle_cid=str(evidence["evidence_root"]),
        admission_receipt_cid=admission_cid,
    )
    predecessor_roots = _revision_roots(
        config,
        population,
        state,
        program_root=predecessor.content_id,
    )
    predecessor_admission_cid = content_identity(
        {
            "operation": "adopt-live-predecessor",
            "predecessor_plan_cid": predecessor.content_id,
            "evidence_root": evidence["evidence_root"],
        }
    )
    goal_cids = tuple(
        [predecessor.goals[0].content_id]
        + [subgoal.content_id for subgoal in predecessor.subgoals]
    )
    retained_cids = tuple(sorted(live_by_cid))
    predecessor_revision = _revision_contract(
        plan_root_cid=predecessor.content_id,
        semantic_revision=1,
        parent_plan_root="",
        origin=PlanOrigin.CREATE,
        roots=predecessor_roots,
        request_cid=predecessor_admission_cid,
        delta_cid="",
        evidence_root=str(evidence["evidence_root"]),
        admission_receipt_cid=predecessor_admission_cid,
        goal_cids=goal_cids,
        task_cids=retained_cids,
        added_cids=retained_cids,
        retained_cids=(),
        claimed_cids=(),
        completed_cids=completed_cids,
        blocked_cids=blocked_cids,
        control_path=control_relative,
        coordination_path=coordination_relative,
    )
    successor_revision = _revision_contract(
        plan_root_cid=str(population["plan_root_cid"]),
        semantic_revision=2,
        parent_plan_root=predecessor.content_id,
        origin=PlanOrigin.STEER,
        roots=roots,
        request_cid=request_cid,
        delta_cid=delta.delta_cid,
        evidence_root=str(evidence["evidence_root"]),
        admission_receipt_cid=admission_cid,
        goal_cids=goal_cids,
        task_cids=tuple(sorted(candidate_by_cid)),
        added_cids=(expected_added_cid,),
        retained_cids=retained_cids,
        claimed_cids=(),
        completed_cids=completed_cids,
        blocked_cids=blocked_cids,
        control_path=control_relative,
        coordination_path=coordination_relative,
    )
    preview = {
        "schema": SUCCESSOR_PREVIEW_SCHEMA,
        "disposition": "admitted",
        "write_performed": False,
        "predecessor_plan_cid": predecessor.content_id,
        "candidate_plan_cid": population["plan_root_cid"],
        "predecessor_revision": predecessor_revision.to_dict(),
        "successor_revision": successor_revision.to_dict(),
        "delta": delta.to_dict(),
        "evidence": evidence,
        "admission": admission_body,
        "candidate_population": candidate_population,
        "candidate_task_spec_cids": {
            alias: str(task.get("spec_cid") or "")
            for alias, task in sorted(candidate_by_alias.items())
        },
        "retained_task_cids": list(retained_cids),
        "completed_task_cids": list(completed_cids),
        "blocked_task_cids": list(blocked_cids),
        "added_task_cids": [expected_added_cid],
        "amended_aliases": sorted(SUCCESSOR_AMENDED_ALIASES),
        "reprioritized_aliases": sorted(SUCCESSOR_REPRIORITIZED_ALIASES),
        "runtime_dependency_edges": admission_body["runtime_dependency_edges"],
        "runtime_dependency_expected_cids": {
            alias: sorted(
                str(dependency.get("dependency_task_cid") or "")
                for dependency in live_by_alias[alias].get("dependencies") or ()
                if isinstance(dependency, Mapping)
            )
            for alias in sorted(SUCCESSOR_RUNTIME_DEPENDENCIES)
        },
        "expected_effects": list(expected_effects),
        "execution_store_sha256": state["store_observations"]["execution"]["sha256"],
    }
    preview["preview_cid"] = content_identity(preview)
    return preview


def _composite_projection(
    control_path: Path, coordination_path: Path
) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    source = DatabaseTaskSource(
        control_path,
        owner_id="lgcvf-successor:projection",
        install_schema=False,
    )
    try:
        plan = _plain_json(source.plan_projection())
        completion = _plain_json(source.completion_evidence_projection())
    finally:
        source.close()
    coordination = read_coordination_registry_projection(coordination_path)
    projection = {
        "schema": SUCCESSOR_COMPOSITE_PROJECTION_SCHEMA,
        "plan_projection_cid": str(plan.get("projection_cid") or ""),
        "completion_projection_cid": str(completion.get("projection_cid") or ""),
        "coordination_projection_root": str(coordination.get("projection_root") or ""),
    }
    projection["projection_cid"] = content_identity(projection)
    return projection


def _validate_applied_successor_projection(
    control_path: Path,
    coordination_path: Path,
    *,
    expected_task_spec_cids: Mapping[str, str],
    completed_task_cids: Sequence[str],
    expected_completion_binding: Mapping[str, Any],
    expected_blocker_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Check protected postconditions while the revision backup is live."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    source = DatabaseTaskSource(
        control_path,
        owner_id="lgcvf-successor:postcondition",
        install_schema=False,
    )
    try:
        plan = _plain_json(source.plan_projection())
        completion = _plain_json(source.completion_evidence_projection())
    finally:
        source.close()
    coordination = read_coordination_registry_projection(coordination_path)
    by_cid, by_alias = _task_map(plan, noun="applied successor projection")
    if set(by_alias) != set(expected_task_spec_cids):
        raise MaterializationError("applied successor task population differs")
    for alias, expected_spec_cid in expected_task_spec_cids.items():
        if by_alias[alias].get("spec_cid") != expected_spec_cid:
            raise MaterializationError(f"{alias}: applied successor spec differs")
    registered = {
        str(item.get("task_cid") or ""): str(item.get("task_id") or "")
        for item in coordination.get("tasks") or ()
        if isinstance(item, Mapping)
    }
    expected_registered = {
        cid: str(task.get("task_alias") or "") for cid, task in by_cid.items()
    }
    if registered != expected_registered:
        raise MaterializationError("applied successor coordination registry differs")
    expected_edges = {
        (cid, str(dependency.get("dependency_task_cid") or ""))
        for cid, task in by_cid.items()
        for dependency in task.get("dependencies") or ()
        if isinstance(dependency, Mapping)
    }
    observed_edges = {
        (str(item.get("task_cid") or ""), str(item.get("dependency_task_cid") or ""))
        for item in coordination.get("dependency_edges") or ()
        if isinstance(item, Mapping)
    }
    if observed_edges != expected_edges:
        raise MaterializationError("applied successor coordination dependencies differ")
    state = {
        "completion_projection": completion,
        "coordination_projection": coordination,
    }
    if _retained_completion_binding(state, completed_task_cids) != dict(
        expected_completion_binding
    ):
        raise MaterializationError(
            "accepted completion evidence changed during successor apply"
        )
    if _protected_blocker_binding(by_alias) != dict(expected_blocker_binding):
        raise MaterializationError(
            "protected blocker authority changed during successor apply"
        )
    projection = {
        "schema": SUCCESSOR_COMPOSITE_PROJECTION_SCHEMA,
        "plan_projection_cid": str(plan.get("projection_cid") or ""),
        "completion_projection_cid": str(completion.get("projection_cid") or ""),
        "coordination_projection_root": str(
            coordination.get("projection_root") or ""
        ),
    }
    projection["projection_cid"] = content_identity(projection)
    return projection


@dataclass
class _SuccessorProjectionAdapter:
    """Narrow adapter joining control and coordination under one rollback set."""

    database_path: Path
    coordination_path: Path
    candidate_population: Mapping[str, Any]
    predecessor_plan_cid: str
    expected_task_spec_cids: Mapping[str, str]
    completed_task_cids: tuple[str, ...]
    expected_completion_binding: Mapping[str, Any]
    expected_blocker_binding: Mapping[str, Any]
    runtime_dependency_expected_cids: Mapping[str, Sequence[str]]

    def plan_revision_projection_paths(self) -> Mapping[str, Path]:
        return {
            "control": self.database_path,
            "coordination": self.coordination_path,
        }

    def plan_revision_projection_cid(self) -> str:
        return str(
            _composite_projection(
                self.database_path, self.coordination_path
            )["projection_cid"]
        )

    def apply_plan_revision(self, **kwargs: Any) -> Mapping[str, Any]:
        from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
            DatabaseCoordinator,
        )
        from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
            DatabaseTaskSource,
        )

        source = DatabaseTaskSource(
            self.database_path,
            owner_id="lgcvf-successor:single-writer",
            plan_root_cid=self.predecessor_plan_cid,
            install_schema=False,
        )
        try:
            result = source.apply_plan_revision(
                **{
                    **kwargs,
                    "goal_graph": self.candidate_population,
                    "admission": None,
                }
            )
            task = source.get_task(SUCCESSOR_ADDED_ALIAS)
            if task is None:
                raise MaterializationError("LGCVF-113 was not added to the control store")
            added_task = task.to_dict()
        finally:
            source.close()
        candidate_by_alias = {
            str(task.get("task_id") or ""): task
            for task in self.candidate_population.get("tasks") or ()
            if isinstance(task, Mapping)
        }
        coordinator = DatabaseCoordinator(self.coordination_path)
        try:
            coordinator.open()
            registration = coordinator.register_task(
                task_cid=str(added_task["task_cid"]),
                task_id=str(added_task["task_alias"]),
                dependency_task_cids=tuple(str(item) for item in added_task["dependencies"]),
                body={
                    "task_alias": added_task["task_alias"],
                    "status": added_task["status"],
                    "priority": added_task["priority"],
                    "authority": "lgcvf-plan-revision-2",
                },
            )
            dependency_amendments: list[dict[str, Any]] = []
            for task_alias, dependency_aliases in sorted(
                SUCCESSOR_RUNTIME_DEPENDENCIES.items()
            ):
                target = candidate_by_alias.get(task_alias)
                if not isinstance(target, Mapping):
                    raise MaterializationError(
                        f"{task_alias}: runtime dependency target is absent"
                    )
                current_expected = [
                    str(item)
                    for item in self.runtime_dependency_expected_cids.get(
                        task_alias, ()
                    )
                ]
                for dependency_alias in dependency_aliases:
                    dependency = candidate_by_alias.get(dependency_alias)
                    if not isinstance(dependency, Mapping):
                        raise MaterializationError(
                            f"{dependency_alias}: runtime dependency task is absent"
                        )
                    dependency_cid = str(dependency.get("task_cid") or "")
                    dependency_amendments.append(
                        coordinator.add_unstarted_task_dependency(
                            task_cid=str(target.get("task_cid") or ""),
                            dependency_task_cid=dependency_cid,
                            expected_dependency_task_cids=tuple(current_expected),
                            operation_id=(
                                "lgcvf-revision-2-dependency:"
                                f"{task_alias}:{dependency_alias}"
                            ),
                        )
                    )
                    current_expected.append(dependency_cid)
        finally:
            coordinator.close()
        projection = _validate_applied_successor_projection(
            self.database_path,
            self.coordination_path,
            expected_task_spec_cids=self.expected_task_spec_cids,
            completed_task_cids=self.completed_task_cids,
            expected_completion_binding=self.expected_completion_binding,
            expected_blocker_binding=self.expected_blocker_binding,
        )
        return {
            **dict(result),
            "coordination_registration": registration,
            "coordination_dependency_amendments": dependency_amendments,
            "projection_cid": projection["projection_cid"],
        }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _load_successor_receipt(path: Path) -> dict[str, Any]:
    try:
        value = _strict_json_loads(path.read_bytes(), noun="successor receipt")
    except (OSError, MaterializationError) as exc:
        raise MaterializationError("successor receipt is absent or unreadable") from exc
    if not isinstance(value, dict) or value.get("schema") != SUCCESSOR_RECEIPT_SCHEMA:
        raise MaterializationError("successor receipt schema differs")
    claimed = str(value.pop("receipt_cid", ""))
    observed = content_identity(value)
    value["receipt_cid"] = claimed
    if not claimed or claimed != observed:
        raise MaterializationError("successor receipt content identity differs")
    return value


def _successor_receipt_path(paths: Mapping[str, Path], revision_cid: str) -> Path:
    if not revision_cid or "/" in revision_cid or ".." in revision_cid:
        raise MaterializationError("successor revision CID is unsafe")
    return paths["revision_receipts"] / f"{revision_cid}.json"


def _successor_manifest_path(paths: Mapping[str, Path], revision_cid: str) -> Path:
    if not revision_cid or "/" in revision_cid or ".." in revision_cid:
        raise MaterializationError("successor revision CID is unsafe")
    return (
        paths["revision_store"]
        / "lgcvf-successor-manifests"
        / f"{revision_cid}.json"
    )


def _load_successor_manifest(path: Path) -> dict[str, Any]:
    try:
        value = _strict_json_loads(
            path.read_bytes(), noun="successor recovery manifest"
        )
    except (OSError, MaterializationError) as exc:
        raise MaterializationError("successor recovery manifest is absent or unreadable") from exc
    if not isinstance(value, dict) or value.get("schema") != SUCCESSOR_RECOVERY_MANIFEST_SCHEMA:
        raise MaterializationError("successor recovery manifest schema differs")
    claimed = str(value.pop("manifest_cid", ""))
    observed = content_identity(value)
    value["manifest_cid"] = claimed
    if not claimed or claimed != observed:
        raise MaterializationError("successor recovery manifest identity differs")
    return value


def _prepare_successor_manifest(
    paths: Mapping[str, Path],
    preview: Mapping[str, Any],
    before: Mapping[str, str],
) -> dict[str, Any]:
    candidate_tasks = {
        str(task.get("task_id") or ""): str(task.get("task_cid") or "")
        for task in preview["candidate_population"].get("tasks") or ()
        if isinstance(task, Mapping)
    }
    manifest = {
        "schema": SUCCESSOR_RECOVERY_MANIFEST_SCHEMA,
        "predecessor_plan_cid": preview["predecessor_plan_cid"],
        "candidate_plan_cid": preview["candidate_plan_cid"],
        "predecessor_revision_cid": content_identity(preview["predecessor_revision"]),
        "successor_revision_cid": content_identity(preview["successor_revision"]),
        "delta_cid": content_identity(preview["delta"]),
        "preview_cid": preview["preview_cid"],
        "predecessor_archive_sha256": preview["evidence"][
            "predecessor_archive_sha256"
        ],
        "bootstrap_receipt_cid": preview["evidence"]["bootstrap_receipt_cid"],
        "predecessor_evidence_root": preview["evidence"]["evidence_root"],
        "retained_task_cids": list(preview["retained_task_cids"]),
        "completed_task_cids": list(preview["completed_task_cids"]),
        "blocked_task_cids": list(preview["blocked_task_cids"]),
        "added_task_cids": list(preview["added_task_cids"]),
        "candidate_task_cids": candidate_tasks,
        "candidate_task_spec_cids": dict(preview["candidate_task_spec_cids"]),
        "retained_completion_binding": preview["evidence"][
            "retained_completion_binding"
        ],
        "protected_blocker_binding": preview["evidence"][
            "protected_blocker_binding"
        ],
        "database_sha256_before": dict(before),
    }
    manifest["manifest_cid"] = content_identity(manifest)
    path = _successor_manifest_path(
        paths, str(manifest["successor_revision_cid"])
    )
    if path.exists():
        existing = _load_successor_manifest(path)
        if existing != manifest:
            raise MaterializationError(
                "existing successor recovery manifest differs from current evidence"
            )
        return existing
    _atomic_write_json(path, manifest)
    return manifest


def _committed_apply_receipt(
    store: Any,
    *,
    revision_cid: str,
    candidate_plan_cid: str,
    delta_cid: str,
) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import (
        PlanRevisionStoreError,
    )

    continuation_paths = sorted(store.continuations_dir.glob("*.json"))
    if len(continuation_paths) > 64:
        raise MaterializationError("plan revision continuation population exceeds bound")
    committed: list[Mapping[str, Any]] = []
    for path in continuation_paths:
        try:
            record = _strict_json_loads(
                path.read_bytes(), noun="plan revision continuation"
            )
        except (OSError, MaterializationError) as exc:
            raise MaterializationError("plan revision continuation is unreadable") from exc
        if not isinstance(record, Mapping):
            raise MaterializationError("plan revision continuation is malformed")
        key = str(record.get("idempotency_key") or "")
        payload = store.load_continuation(key) if key else None
        if (
            isinstance(payload, Mapping)
            and payload.get("phase") == "committed"
            and payload.get("revision_cid") == revision_cid
        ):
            committed.append(payload)
    if len(committed) != 1:
        raise MaterializationError(
            "exactly one committed successor continuation is required"
        )
    receipt_cid = str(committed[0].get("receipt_cid") or "")
    try:
        receipt = store.get_cas(receipt_cid)
    except (OSError, PlanRevisionStoreError) as exc:
        raise MaterializationError("committed plan-revision receipt is unavailable") from exc
    if (
        receipt.get("receipt_cid") != receipt_cid
        or receipt.get("state") != "committed"
        or receipt.get("revision_cid") != revision_cid
        or receipt.get("plan_root_cid") != candidate_plan_cid
        or receipt.get("delta_cid") != delta_cid
    ):
        raise MaterializationError("committed plan-revision receipt identity differs")
    return _plain_json(receipt)


def _build_successor_receipt(
    manifest: Mapping[str, Any],
    apply_receipt: Mapping[str, Any],
    post_state: Mapping[str, Any],
    after: Mapping[str, str],
) -> dict[str, Any]:
    receipt = {
        "schema": SUCCESSOR_RECEIPT_SCHEMA,
        "authority_mode": "embedded-single-writer",
        "production_authorized": False,
        "predecessor_plan_cid": manifest["predecessor_plan_cid"],
        "candidate_plan_cid": manifest["candidate_plan_cid"],
        "predecessor_revision_cid": manifest["predecessor_revision_cid"],
        "successor_revision_cid": manifest["successor_revision_cid"],
        "delta_cid": manifest["delta_cid"],
        "preview_cid": manifest["preview_cid"],
        "recovery_manifest_cid": manifest["manifest_cid"],
        "plan_revision_apply_receipt": dict(apply_receipt),
        "predecessor_archive_sha256": manifest["predecessor_archive_sha256"],
        "bootstrap_receipt_cid": manifest["bootstrap_receipt_cid"],
        "bootstrap_receipt_sha256": manifest["database_sha256_before"]["receipt"],
        "predecessor_evidence_root": manifest["predecessor_evidence_root"],
        "retained_task_cids": list(manifest["retained_task_cids"]),
        "completed_task_cids": list(manifest["completed_task_cids"]),
        "blocked_task_cids": list(manifest["blocked_task_cids"]),
        "added_task_cids": list(manifest["added_task_cids"]),
        "candidate_task_spec_cids": dict(manifest["candidate_task_spec_cids"]),
        "retained_completion_binding": manifest["retained_completion_binding"],
        "protected_blocker_binding": manifest["protected_blocker_binding"],
        "post_composite_projection": post_state["composite_projection"],
        "database_sha256_before": dict(manifest["database_sha256_before"]),
        "database_sha256_after": dict(after),
        "execution_store_mutated": False,
        "bootstrap_receipt_mutated": False,
        "historical_status_rewritten": False,
        "manual_or_external_task_completed": False,
    }
    receipt["receipt_cid"] = content_identity(receipt)
    return receipt


def _finalize_committed_successor(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    root: Path,
    paths: Mapping[str, Path],
    store: Any,
) -> dict[str, Any]:
    """Reconstruct the external receipt after a committed crash window."""

    predecessor = _load_predecessor_plan(config, root=root)
    active = store.get_active()
    candidate_plan_cid = str(population.get("plan_root_cid") or "")
    if (
        active is None
        or active.plan_root_cid != candidate_plan_cid
        or active.semantic_revision != 2
        or active.quarantined
    ):
        raise MaterializationError("committed successor recovery has no valid active head")
    revision = store.load_revision(active.revision_cid)
    if revision.parent_plan_root != predecessor.content_id:
        raise MaterializationError("committed successor ancestry differs")
    manifest = _load_successor_manifest(
        _successor_manifest_path(paths, active.revision_cid)
    )
    if (
        manifest.get("predecessor_plan_cid") != predecessor.content_id
        or manifest.get("candidate_plan_cid") != candidate_plan_cid
        or manifest.get("successor_revision_cid") != active.revision_cid
        or manifest.get("delta_cid") != revision.delta_cid
        or manifest.get("predecessor_archive_sha256")
        != _sha256_file(paths["predecessor_archive"])
    ):
        raise MaterializationError("committed successor recovery manifest is stale")
    apply_receipt = _committed_apply_receipt(
        store,
        revision_cid=active.revision_cid,
        candidate_plan_cid=candidate_plan_cid,
        delta_cid=revision.delta_cid,
    )
    post_state = _read_successor_state(config, root=root)
    live_by_cid, live_by_alias = _task_map(
        post_state["plan_projection"], noun="committed successor projection"
    )
    expected_task_cids = manifest.get("candidate_task_cids")
    if not isinstance(expected_task_cids, Mapping) or {
        alias: str(task.get("task_cid") or "")
        for alias, task in live_by_alias.items()
    } != dict(expected_task_cids):
        raise MaterializationError("committed successor logical task identities differ")
    expected_specs = manifest.get("candidate_task_spec_cids")
    if not isinstance(expected_specs, Mapping):
        raise MaterializationError("committed successor spec manifest is malformed")
    projection = _validate_applied_successor_projection(
        paths["control"],
        paths["coordination"],
        expected_task_spec_cids={str(k): str(v) for k, v in expected_specs.items()},
        completed_task_cids=tuple(
            str(item) for item in manifest.get("completed_task_cids") or ()
        ),
        expected_completion_binding=manifest["retained_completion_binding"],
        expected_blocker_binding=manifest["protected_blocker_binding"],
    )
    if (
        projection != post_state["composite_projection"]
        or projection["projection_cid"] != apply_receipt["duckdb_projection_cid"]
        or len(live_by_cid) != len(expected_task_cids)
    ):
        raise MaterializationError("committed successor projection receipt differs")
    before = manifest.get("database_sha256_before")
    if not isinstance(before, Mapping):
        raise MaterializationError("committed successor pre-state hashes are absent")
    after = {
        key: _sha256_file(paths[key])
        for key in ("control", "coordination", "execution", "receipt")
    }
    if after["execution"] != before.get("execution"):
        raise MaterializationError("committed successor changed the execution store")
    if after["receipt"] != before.get("receipt"):
        raise MaterializationError("committed successor changed the bootstrap receipt")
    receipt = _build_successor_receipt(manifest, apply_receipt, post_state, after)
    receipt_path = _successor_receipt_path(paths, active.revision_cid)
    _atomic_write_json(receipt_path, receipt)
    verify_successor_read_only(config, population, root=root)
    return receipt


def steer_successor(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    root: Path = ROOT,
    fault_injector: Any | None = None,
) -> dict[str, Any]:
    """Adopt revision 1 and atomically steer control+coordination to revision 2."""

    if _targets_fresh_recovery_generation(config, root=root):
        raise MaterializationError(
            "fresh recovery targets reject legacy successor mutation"
        )

    from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import (
        PlanRevisionApplyRequest,
        PlanRevisionStore,
        PlanRevisionStoreError,
    )

    paths = _successor_paths(config, root=root)
    try:
        store = PlanRevisionStore(paths["revision_store"], recover=True)
    except PlanRevisionStoreError as exc:
        raise MaterializationError(
            f"plan revision recovery failed closed: {exc}"
        ) from exc
    existing_receipts = sorted(paths["revision_receipts"].glob("*.json"))
    if existing_receipts:
        if len(existing_receipts) != 1:
            raise MaterializationError("multiple successor receipts fail idempotent replay")
        receipt = _load_successor_receipt(existing_receipts[0])
        verify_successor_read_only(config, population, root=root)
        return receipt
    active = store.get_active()
    candidate_plan_cid = str(population.get("plan_root_cid") or "")
    if (
        active is not None
        and active.plan_root_cid == candidate_plan_cid
        and active.semantic_revision == 2
    ):
        return _finalize_committed_successor(
            config,
            population,
            root=root,
            paths=paths,
            store=store,
        )
    preview = preview_successor(config, population, root=root)
    predecessor_revision = PlanRevision.from_dict(preview["predecessor_revision"])
    successor_revision = PlanRevision.from_dict(preview["successor_revision"])
    delta = PlanDelta.from_dict(preview["delta"])
    receipt_path = _successor_receipt_path(paths, successor_revision.revision_cid)
    if receipt_path.exists():
        raise MaterializationError("successor receipt path collided after preview")

    before = {
        key: _sha256_file(paths[key])
        for key in ("control", "coordination", "execution", "receipt")
    }
    active = store.get_active()
    if active is None:
        adoption_base = (
            "lgcvf-adopt-revision-1:"
            + str(preview["evidence"]["evidence_root"])
        )
        adoption_key = ""
        for retry_index in range(9):
            candidate_key = (
                adoption_base
                if retry_index == 0
                else f"{adoption_base}:retry-{retry_index}"
            )
            continuation = store.load_continuation(candidate_key)
            if continuation is None:
                adoption_key = candidate_key
                break
            if str(continuation.get("phase") or "") not in {"restored", "blocked"}:
                raise MaterializationError(
                    "prior predecessor adoption remains non-retryable"
                )
        if not adoption_key:
            raise MaterializationError("predecessor adoption retry key budget is exhausted")
        try:
            adoption = store.apply(
                PlanRevisionApplyRequest(
                    revision=predecessor_revision,
                    observed_roots=predecessor_revision.roots,
                    idempotency_key=adoption_key,
                    expected_effects=("adopt-live-predecessor",),
                    records={
                        "predecessor-formal-plan": {
                            "plan_cid": preview["predecessor_plan_cid"],
                            "archive_sha256": preview["evidence"][
                                "predecessor_archive_sha256"
                            ],
                        },
                        "predecessor-evidence": preview["evidence"],
                    },
                )
            )
        except PlanRevisionStoreError as exc:
            raise MaterializationError(
                f"predecessor revision adoption failed closed: {exc}"
            ) from exc
        active = store.get_active()
        if active is None or not adoption.committed:
            raise MaterializationError("predecessor adoption did not commit")
    if (
        active.plan_root_cid != predecessor_revision.plan_root_cid
        or active.revision_cid != predecessor_revision.revision_cid
        or active.semantic_revision != 1
    ):
        raise MaterializationError("revision store active pointer is not the predecessor")
    manifest = _prepare_successor_manifest(paths, preview, before)

    adapter = _SuccessorProjectionAdapter(
        database_path=paths["control"],
        coordination_path=paths["coordination"],
        candidate_population=preview["candidate_population"],
        predecessor_plan_cid=predecessor_revision.plan_root_cid,
        expected_task_spec_cids=preview["candidate_task_spec_cids"],
        completed_task_cids=tuple(preview["completed_task_cids"]),
        expected_completion_binding=preview["evidence"][
            "retained_completion_binding"
        ],
        expected_blocker_binding=preview["evidence"]["protected_blocker_binding"],
        runtime_dependency_expected_cids=preview[
            "runtime_dependency_expected_cids"
        ],
    )
    idempotency_base = f"lgcvf-steer-revision-2:{preview['preview_cid']}"
    idempotency_key = ""
    for retry_index in range(9):
        candidate_key = (
            idempotency_base
            if retry_index == 0
            else f"{idempotency_base}:retry-{retry_index}"
        )
        continuation = store.load_continuation(candidate_key)
        if continuation is None:
            idempotency_key = candidate_key
            break
        phase = str(continuation.get("phase") or "")
        if phase not in {"restored", "blocked"}:
            raise MaterializationError(
                f"prior successor continuation remains non-retryable at {phase!r}"
            )
    if not idempotency_key:
        raise MaterializationError("successor retry key budget is exhausted")
    try:
        apply_receipt = store.apply(
            PlanRevisionApplyRequest(
                revision=successor_revision,
                observed_roots=successor_revision.roots,
                idempotency_key=idempotency_key,
                expected_effects=tuple(preview["expected_effects"]),
                delta=delta,
                goal_graph=preview["candidate_population"],
                duckdb_source=adapter,
                repository_tree_id=str(
                    preview["evidence"]["base_repository_tree_id"]
                ),
                fencing_token=1,
                base_event_cursor=active.event_cursor,
                expected_active_plan_root=active.plan_root_cid,
                expected_active_revision_cid=active.revision_cid,
                fault_injector=fault_injector,
                records={
                    "successor-formal-plan": {
                        "plan_cid": preview["candidate_plan_cid"],
                        "population_root": population["population_root"],
                    },
                    "successor-admission": preview["admission"],
                    "successor-recovery-manifest": manifest,
                },
            )
        )
    except PlanRevisionStoreError as exc:
        restored = {
            key: _sha256_file(paths[key])
            for key in ("control", "coordination", "execution", "receipt")
        }
        if restored != before:
            raise MaterializationError(
                "successor apply failed and exact operational rollback did not verify"
            ) from exc
        raise MaterializationError(
            f"successor apply failed after exact operational rollback: {exc}"
        ) from exc
    if not apply_receipt.committed:
        raise MaterializationError("successor plan revision did not commit")
    if callable(fault_injector):
        fault_injector("after_revision_commit_before_external_receipt")
    after = {
        key: _sha256_file(paths[key])
        for key in ("control", "coordination", "execution", "receipt")
    }
    if after["execution"] != before["execution"]:
        raise MaterializationError("successor apply changed the execution store")
    if after["receipt"] != before["receipt"]:
        raise MaterializationError("successor apply changed the bootstrap receipt")
    post_state = _read_successor_state(config, root=root)
    post_plan_by_cid, post_plan_by_alias = _task_map(
        post_state["plan_projection"], noun="applied successor projection"
    )
    if set(post_plan_by_alias) != set(preview["candidate_task_spec_cids"]):
        raise MaterializationError("applied successor task population differs")
    for alias, expected_spec in preview["candidate_task_spec_cids"].items():
        if post_plan_by_alias[alias].get("spec_cid") != expected_spec:
            raise MaterializationError(f"{alias}: applied successor spec differs")
    predecessor_completed = set(preview["completed_task_cids"])
    post_completion_binding = _retained_completion_binding(
        post_state, sorted(predecessor_completed)
    )
    if post_completion_binding != preview["evidence"]["retained_completion_binding"]:
        raise MaterializationError("accepted completion evidence changed during successor apply")
    post_blocker_binding = _protected_blocker_binding(post_plan_by_alias)
    if post_blocker_binding != preview["evidence"]["protected_blocker_binding"]:
        raise MaterializationError("protected blocker authority changed during successor apply")
    receipt = _build_successor_receipt(
        manifest, apply_receipt.to_dict(), post_state, after
    )
    _atomic_write_json(receipt_path, receipt)
    return receipt


def _directory_fingerprint(
    path: Path, *, require_private: bool = False
) -> dict[str, dict[str, Any]]:
    """Fingerprint a closed, owner-controlled tree without following links."""

    common_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags = (
        common_flags | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        root_descriptor = os.open(path, directory_flags)
    except OSError as exc:
        raise MaterializationError(f"required directory is absent: {path}") from exc
    try:
        root_status = os.fstat(root_descriptor)
        if (
            not stat.S_ISDIR(root_status.st_mode)
            or root_status.st_uid != os.geteuid()
            or (
                require_private
                and (
                    stat.S_IMODE(root_status.st_mode) & 0o022
                    or stat.S_IMODE(root_status.st_mode) & 0o700 != 0o700
                )
            )
        ):
            raise MaterializationError("authority directory permissions differ")
        result: dict[str, dict[str, Any]] = {
            ".": {
                "kind": "directory",
                "mode": stat.S_IMODE(root_status.st_mode),
                "uid": root_status.st_uid,
                "dev": root_status.st_dev,
                "ino": root_status.st_ino,
            }
        }
        count = 0

        def walk(directory_descriptor: int, prefix: str) -> None:
            nonlocal count
            for name in sorted(os.listdir(directory_descriptor)):
                count += 1
                if count > 10_000:
                    raise MaterializationError(
                        "authority tree population exceeds bound"
                    )
                relative = f"{prefix}/{name}" if prefix else name
                before = os.stat(
                    name, dir_fd=directory_descriptor, follow_symlinks=False
                )
                if stat.S_ISDIR(before.st_mode):
                    child = os.open(
                        name, directory_flags, dir_fd=directory_descriptor
                    )
                    try:
                        opened = os.fstat(child)
                        if (
                            opened.st_uid != os.geteuid()
                            or (
                                require_private
                                and (
                                    stat.S_IMODE(opened.st_mode) & 0o022
                                    or stat.S_IMODE(opened.st_mode) & 0o700
                                    != 0o700
                                )
                            )
                            or (before.st_dev, before.st_ino)
                            != (opened.st_dev, opened.st_ino)
                        ):
                            raise MaterializationError(
                                "authority directory identity differs"
                            )
                        result[relative + "/"] = {
                            "kind": "directory",
                            "mode": stat.S_IMODE(opened.st_mode),
                            "uid": opened.st_uid,
                            "dev": opened.st_dev,
                            "ino": opened.st_ino,
                        }
                        walk(child, relative)
                        after = os.fstat(child)
                        if (
                            opened.st_dev,
                            opened.st_ino,
                            opened.st_mode,
                        ) != (after.st_dev, after.st_ino, after.st_mode):
                            raise MaterializationError(
                                "authority directory changed while fingerprinting"
                            )
                    finally:
                        os.close(child)
                elif stat.S_ISREG(before.st_mode):
                    child = os.open(
                        name,
                        common_flags | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=directory_descriptor,
                    )
                    try:
                        opened = os.fstat(child)
                        if (
                            opened.st_uid != os.geteuid()
                            or opened.st_nlink != 1
                            or (
                                require_private
                                and (
                                    stat.S_IMODE(opened.st_mode) & 0o022
                                    or stat.S_IMODE(opened.st_mode) & 0o400
                                    != 0o400
                                )
                            )
                            or (before.st_dev, before.st_ino)
                            != (opened.st_dev, opened.st_ino)
                            or opened.st_size > 256 * 1024 * 1024
                        ):
                            raise MaterializationError(
                                "authority file identity differs"
                            )
                        digest = hashlib.sha256()
                        observed = 0
                        while True:
                            chunk = os.read(child, 1024 * 1024)
                            if not chunk:
                                break
                            observed += len(chunk)
                            if observed > 256 * 1024 * 1024:
                                raise MaterializationError(
                                    "authority file exceeds bound"
                                )
                            digest.update(chunk)
                        after = os.fstat(child)
                        if (
                            opened.st_dev,
                            opened.st_ino,
                            opened.st_mode,
                            opened.st_size,
                            opened.st_mtime_ns,
                        ) != (
                            after.st_dev,
                            after.st_ino,
                            after.st_mode,
                            after.st_size,
                            after.st_mtime_ns,
                        ) or observed != opened.st_size:
                            raise MaterializationError(
                                "authority file changed while fingerprinting"
                            )
                        result[relative] = {
                            "kind": "file",
                            "mode": stat.S_IMODE(opened.st_mode),
                            "uid": opened.st_uid,
                            "dev": opened.st_dev,
                            "ino": opened.st_ino,
                            "size": opened.st_size,
                            "mtime_ns": opened.st_mtime_ns,
                            "sha256": "sha256:" + digest.hexdigest(),
                        }
                    finally:
                        os.close(child)
                else:
                    raise MaterializationError(
                        "authority tree contains a link or special file"
                    )

        walk(root_descriptor, "")
        after_root = os.fstat(root_descriptor)
        if (
            root_status.st_dev,
            root_status.st_ino,
            root_status.st_mode,
        ) != (after_root.st_dev, after_root.st_ino, after_root.st_mode):
            raise MaterializationError(
                "authority root changed while fingerprinting"
            )
        return result
    except OSError as exc:
        raise MaterializationError("authority tree cannot be fingerprinted") from exc
    finally:
        os.close(root_descriptor)


def verify_successor_read_only(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Reconstruct revision ancestry and immutable task specs without writes."""

    if _targets_fresh_recovery_generation(config, root=root):
        raise MaterializationError(
            "fresh recovery targets reject legacy successor verification"
        )

    from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import (
        PlanRevisionStore,
    )

    predecessor = _load_predecessor_plan(config, root=root)
    paths = _successor_paths(config, root=root)
    if not paths["revision_store"].is_dir():
        raise MaterializationError("plan revision store is absent")
    receipt_files = sorted(paths["revision_receipts"].glob("*.json"))
    if len(receipt_files) != 1:
        raise MaterializationError("exactly one successor revision receipt is required")
    receipt = _load_successor_receipt(receipt_files[0])
    manifest = _load_successor_manifest(
        _successor_manifest_path(
            paths, str(receipt.get("successor_revision_cid") or "")
        )
    )
    if (
        receipt.get("recovery_manifest_cid") != manifest.get("manifest_cid")
        or receipt.get("preview_cid") != manifest.get("preview_cid")
        or receipt.get("delta_cid") != manifest.get("delta_cid")
        or receipt.get("database_sha256_before")
        != manifest.get("database_sha256_before")
        or receipt.get("retained_completion_binding")
        != manifest.get("retained_completion_binding")
        or receipt.get("protected_blocker_binding")
        != manifest.get("protected_blocker_binding")
    ):
        raise MaterializationError("successor receipt/recovery manifest binding differs")
    before_revision_store = _directory_fingerprint(paths["revision_store"])
    before_receipt = (
        receipt_files[0].stat().st_size,
        receipt_files[0].stat().st_mtime_ns,
        _sha256_file(receipt_files[0]),
    )
    state = _read_successor_state(config, root=root)
    plan = state["plan_projection"]
    coordination = state["coordination_projection"]
    active_plans = [
        item
        for item in plan.get("plans") or ()
        if isinstance(item, Mapping) and item.get("status") == "active"
    ]
    candidate_plan_cid = str(population.get("plan_root_cid") or "")
    if len(active_plans) != 1 or active_plans[0].get("plan_cid") != candidate_plan_cid:
        raise MaterializationError("revision-2 plan is not the exact active control head")
    plan_rows = {
        str(item.get("plan_cid") or ""): item
        for item in plan.get("plans") or ()
        if isinstance(item, Mapping)
    }
    if (
        predecessor.content_id not in plan_rows
        or plan_rows[predecessor.content_id].get("status") == "active"
    ):
        raise MaterializationError("predecessor plan was lost or remains active")
    live_by_cid, live_by_alias = _task_map(plan, noun="live successor projection")
    desired_tasks = {
        str(item.get("task_id") or ""): str(item.get("task_cid") or "")
        for item in population.get("tasks") or ()
        if isinstance(item, Mapping)
    }
    expected_tasks = {task.task_id: task.content_id for task in predecessor.tasks}
    if SUCCESSOR_ADDED_ALIAS not in desired_tasks:
        raise MaterializationError("revision-2 formal plan omits LGCVF-113")
    expected_tasks[SUCCESSOR_ADDED_ALIAS] = desired_tasks[SUCCESSOR_ADDED_ALIAS]
    observed_tasks = {
        alias: str(item.get("task_cid") or "") for alias, item in live_by_alias.items()
    }
    if observed_tasks != expected_tasks:
        raise MaterializationError("successor logical task identities differ")
    expected_specs = receipt.get("candidate_task_spec_cids")
    if not isinstance(expected_specs, Mapping) or set(expected_specs) != set(live_by_alias):
        raise MaterializationError("successor receipt task spec population differs")
    revision_histories = state.get("task_revision_histories")
    if not isinstance(revision_histories, Mapping) or set(revision_histories) != set(
        live_by_cid
    ):
        raise MaterializationError("successor task revision history population differs")
    for alias, task in live_by_alias.items():
        try:
            _assert_receipt_bound_task_spec(
                task,
                expected_legacy_spec_cid=str(expected_specs.get(alias) or ""),
                revision_history=revision_histories[str(task.get("task_cid") or "")],
            )
        except MaterializationError as exc:
            raise MaterializationError(
                f"{alias}: current task specification is stale ({exc})"
            ) from exc

    registered = {
        str(item.get("task_cid") or ""): str(item.get("task_id") or "")
        for item in coordination.get("tasks") or ()
        if isinstance(item, Mapping)
    }
    if registered != {cid: alias for alias, cid in expected_tasks.items()}:
        raise MaterializationError("successor coordination task registry differs")
    expected_edges = {
        (cid, str(dependency.get("dependency_task_cid") or ""))
        for cid, task in live_by_cid.items()
        for dependency in task.get("dependencies") or ()
        if isinstance(dependency, Mapping)
    }
    observed_edges = {
        (str(item.get("task_cid") or ""), str(item.get("dependency_task_cid") or ""))
        for item in coordination.get("dependency_edges") or ()
        if isinstance(item, Mapping)
    }
    if observed_edges != expected_edges:
        raise MaterializationError("successor coordination dependencies differ")
    for alias, authority, completion_mode in (
        ("LGCVF-121", "blocked_external_authority", "external-authority"),
        ("LGCVF-123", "blocked_manual", "manual"),
    ):
        task = live_by_alias.get(alias)
        body = task.get("body") if isinstance(task, Mapping) else None
        if (
            not isinstance(task, Mapping)
            or task.get("status") != "blocked"
            or not isinstance(body, Mapping)
            or body.get("construction_status") != authority
            or body.get("completion") != completion_mode
            or body.get("review_only") is not True
        ):
            raise MaterializationError(f"{alias}: protected authority was rewritten")
    retained_completion = receipt.get("retained_completion_binding")
    if not isinstance(retained_completion, Mapping) or (
        _retained_completion_binding(
            state, tuple(str(item) for item in receipt.get("completed_task_cids") or ())
        )
        != retained_completion
    ):
        raise MaterializationError("accepted predecessor completion evidence changed")
    protected_blockers = receipt.get("protected_blocker_binding")
    if not isinstance(protected_blockers, Mapping) or (
        _protected_blocker_binding(live_by_alias) != protected_blockers
    ):
        raise MaterializationError("protected blocker authority binding changed")

    if _sha256_file(paths["receipt"]) != receipt.get("bootstrap_receipt_sha256"):
        raise MaterializationError("bootstrap receipt bytes changed after successor apply")
    bootstrap = state["bootstrap_receipt"]
    if bootstrap.get("receipt_cid") != receipt.get("bootstrap_receipt_cid"):
        raise MaterializationError("bootstrap receipt identity changed")
    store = PlanRevisionStore(paths["revision_store"], recover=False)
    active = store.get_active()
    if (
        active is None
        or active.plan_root_cid != candidate_plan_cid
        or active.revision_cid != receipt.get("successor_revision_cid")
        or active.semantic_revision != 2
        or active.quarantined
    ):
        raise MaterializationError("plan revision store active pointer differs")
    revision = store.load_revision(active.revision_cid)
    if (
        revision.parent_plan_root != predecessor.content_id
        or revision.delta_cid != receipt.get("delta_cid")
        or set(revision.retained_population.member_cids)
        != set(receipt.get("retained_task_cids") or ())
        or set(revision.added_population.member_cids)
        != set(receipt.get("added_task_cids") or ())
    ):
        raise MaterializationError("stored successor ancestry/population differs")
    apply_receipt = receipt.get("plan_revision_apply_receipt")
    if (
        not isinstance(apply_receipt, Mapping)
        or apply_receipt.get("committed") is not True
        or apply_receipt.get("revision_cid") != active.revision_cid
        or apply_receipt.get("plan_root_cid") != active.plan_root_cid
        or apply_receipt.get("delta_cid") != revision.delta_cid
    ):
        raise MaterializationError("plan revision apply receipt differs")
    after_revision_store = _directory_fingerprint(paths["revision_store"])
    after_receipt = (
        receipt_files[0].stat().st_size,
        receipt_files[0].stat().st_mtime_ns,
        _sha256_file(receipt_files[0]),
    )
    if before_revision_store != after_revision_store or before_receipt != after_receipt:
        raise MaterializationError("read-only successor verification changed evidence")
    result = {
        "schema": SUCCESSOR_VERIFICATION_SCHEMA,
        "valid": True,
        "verification_mode": "read_only",
        "predecessor_plan_cid": predecessor.content_id,
        "candidate_plan_cid": candidate_plan_cid,
        "successor_revision_cid": active.revision_cid,
        "delta_cid": revision.delta_cid,
        "successor_receipt_cid": receipt["receipt_cid"],
        "task_count": len(live_by_cid),
        "retained_task_count": len(revision.retained_population.member_cids),
        "added_task_count": len(revision.added_population.member_cids),
        "accepted_history_preserved": True,
        "protected_blockers_preserved": True,
        "execution_store_mutated": False,
        "stores_unchanged": True,
        "active_coordination_counts": {
            key: value
            for key, value in dict(coordination.get("counts") or {}).items()
            if key.startswith("active_")
        },
    }
    result["verification_root"] = content_identity(result)
    return result


def _duckdb_json_value(value: Any) -> Any:
    """Project DuckDB catalog/data values without lossy float identities."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return {"type": "float", "value": repr(value)}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": "bytes", "hex": bytes(value).hex()}
    if isinstance(value, Mapping):
        return {
            str(key): _duckdb_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_duckdb_json_value(item) for item in value]
    if hasattr(value, "isoformat"):
        return {"type": type(value).__name__, "value": str(value.isoformat())}
    return {"type": type(value).__name__, "value": str(value)}


def _duckdb_catalog_projection(connection: Any) -> dict[str, Any]:
    """Return the exact persistent user catalog across every DuckDB schema."""

    database_rows = connection.execute(
        "SELECT comment, tags, internal, type, readonly, encrypted, cipher, options "
        "FROM duckdb_databases() WHERE database_name = current_database()"
    ).fetchall()
    if len(database_rows) != 1:
        raise MaterializationError("DuckDB authority database inventory differs")
    database_row = database_rows[0]
    database = {
        "comment": str(database_row[0] or ""),
        "tags": _duckdb_json_value(database_row[1]),
        "internal": bool(database_row[2]),
        "type": str(database_row[3] or ""),
        "readonly": bool(database_row[4]),
        "encrypted": bool(database_row[5]),
        "cipher": str(database_row[6] or ""),
        "options": _duckdb_json_value(database_row[7]),
    }
    schemas = [
        {
            "name": str(row[0]),
            "comment": str(row[1] or ""),
            "tags": _duckdb_json_value(row[2]),
            "sql": str(row[3] or ""),
        }
        for row in connection.execute(
            "SELECT schema_name, comment, tags, sql FROM duckdb_schemas() "
            "WHERE database_name = current_database() AND NOT internal "
            "ORDER BY schema_name"
        ).fetchall()
    ]
    tables = [
        {
            "schema": str(row[0]),
            "name": str(row[1]),
            "comment": str(row[2] or ""),
            "tags": _duckdb_json_value(row[3]),
            "temporary": bool(row[4]),
            "has_primary_key": bool(row[5]),
            "column_count": int(row[6]),
            "index_count": int(row[7]),
            "check_constraint_count": int(row[8]),
            "sql": str(row[9] or ""),
        }
        for row in connection.execute(
            "SELECT schema_name, table_name, comment, tags, temporary, "
            "has_primary_key, column_count, index_count, "
            "check_constraint_count, sql FROM duckdb_tables() "
            "WHERE database_name = current_database() AND NOT internal "
            "ORDER BY schema_name, table_name"
        ).fetchall()
    ]
    views = [
        {
            "schema": str(row[0]),
            "name": str(row[1]),
            "comment": str(row[2] or ""),
            "tags": _duckdb_json_value(row[3]),
            "temporary": bool(row[4]),
            "column_count": int(row[5]),
            "sql": str(row[6] or ""),
            "bound": bool(row[7]),
        }
        for row in connection.execute(
            "SELECT schema_name, view_name, comment, tags, temporary, "
            "column_count, sql, is_bound FROM duckdb_views() "
            "WHERE database_name = current_database() AND NOT internal "
            "ORDER BY schema_name, view_name"
        ).fetchall()
    ]
    relations = sorted(
        [
            {"schema": item["schema"], "name": item["name"], "type": kind}
            for kind, members in (("BASE TABLE", tables), ("VIEW", views))
            for item in members
        ],
        key=lambda item: (item["schema"], item["name"], item["type"]),
    )
    columns = [
        {
            "schema": str(row[0]),
            "table": str(row[1]),
            "ordinal": int(row[2]),
            "column": str(row[3]),
            "comment": str(row[4] or ""),
            "default": str(row[5] or ""),
            "nullable": bool(row[6]),
            "type": str(row[7]),
            "character_maximum_length": (
                None if row[8] is None else int(row[8])
            ),
            "numeric_precision": None if row[9] is None else int(row[9]),
            "numeric_precision_radix": (
                None if row[10] is None else int(row[10])
            ),
            "numeric_scale": None if row[11] is None else int(row[11]),
        }
        for row in connection.execute(
            "SELECT schema_name, table_name, column_index, column_name, "
            "comment, column_default, is_nullable, data_type, "
            "character_maximum_length, numeric_precision, "
            "numeric_precision_radix, numeric_scale FROM duckdb_columns() "
            "WHERE database_name = current_database() AND NOT internal "
            "ORDER BY schema_name, table_name, column_index"
        ).fetchall()
    ]
    constraints = [
        {
            "schema": str(row[0]),
            "table": str(row[1]),
            "ordinal": int(row[2]),
            "type": str(row[3]),
            "text": str(row[4] or ""),
            "expression": str(row[5] or ""),
            "column_indexes": _duckdb_json_value(row[6]),
            "column_names": _duckdb_json_value(row[7]),
            "name": str(row[8] or ""),
            "referenced_table": str(row[9] or ""),
            "referenced_columns": _duckdb_json_value(row[10]),
        }
        for row in connection.execute(
            "SELECT schema_name, table_name, constraint_index, constraint_type, "
            "constraint_text, expression, constraint_column_indexes, "
            "constraint_column_names, constraint_name, referenced_table, "
            "referenced_column_names FROM duckdb_constraints() "
            "WHERE database_name = current_database() "
            "ORDER BY schema_name, table_name, constraint_index, constraint_name"
        ).fetchall()
    ]
    indexes = [
        {
            "schema": str(row[0]),
            "name": str(row[1]),
            "table": str(row[2]),
            "comment": str(row[3] or ""),
            "tags": _duckdb_json_value(row[4]),
            "unique": bool(row[5]),
            "primary": bool(row[6]),
            "expressions": str(row[7] or ""),
            "sql": str(row[8] or ""),
        }
        for row in connection.execute(
            "SELECT schema_name, index_name, table_name, comment, tags, "
            "is_unique, is_primary, expressions, sql FROM duckdb_indexes() "
            "WHERE database_name = current_database() "
            "ORDER BY schema_name, index_name, table_name"
        ).fetchall()
    ]
    sequences = [
        {
            "schema": str(row[0]),
            "name": str(row[1]),
            "comment": str(row[2] or ""),
            "tags": _duckdb_json_value(row[3]),
            "temporary": bool(row[4]),
            "start": int(row[5]),
            "minimum": int(row[6]),
            "maximum": int(row[7]),
            "increment": int(row[8]),
            "cycle": bool(row[9]),
            "last": None if row[10] is None else int(row[10]),
            "sql": str(row[11] or ""),
        }
        for row in connection.execute(
            "SELECT schema_name, sequence_name, comment, tags, temporary, "
            "start_value, min_value, max_value, increment_by, cycle, "
            "last_value, sql FROM duckdb_sequences() "
            "WHERE database_name = current_database() "
            "ORDER BY schema_name, sequence_name"
        ).fetchall()
    ]
    types = [
        {
            "schema": str(row[0]),
            "name": str(row[1]),
            "size": int(row[2]),
            "logical_type": str(row[3]),
            "category": str(row[4] or ""),
            "comment": str(row[5] or ""),
            "tags": _duckdb_json_value(row[6]),
            "labels": _duckdb_json_value(row[7]),
        }
        for row in connection.execute(
            "SELECT schema_name, type_name, type_size, logical_type, "
            "type_category, comment, tags, labels FROM duckdb_types() "
            "WHERE database_name = current_database() AND NOT internal "
            "ORDER BY schema_name, type_name"
        ).fetchall()
    ]
    macros = [
        {
            "schema": str(row[0]),
            "name": str(row[1]),
            "alias_of": str(row[2] or ""),
            "function_type": str(row[3]),
            "description": str(row[4] or ""),
            "comment": str(row[5] or ""),
            "tags": _duckdb_json_value(row[6]),
            "return_type": str(row[7] or ""),
            "parameters": _duckdb_json_value(row[8]),
            "parameter_types": _duckdb_json_value(row[9]),
            "varargs": str(row[10] or ""),
            "definition": str(row[11] or ""),
            "has_side_effects": (
                None if row[12] is None else bool(row[12])
            ),
            "examples": _duckdb_json_value(row[13]),
            "stability": str(row[14] or ""),
            "categories": _duckdb_json_value(row[15]),
        }
        for row in connection.execute(
            "SELECT schema_name, function_name, alias_of, function_type, "
            "description, comment, tags, return_type, parameters, "
            "parameter_types, varargs, macro_definition, has_side_effects, "
            "examples, stability, categories FROM duckdb_functions() "
            "WHERE database_name = current_database() AND NOT internal "
            "ORDER BY schema_name, function_name, function_type"
        ).fetchall()
    ]
    projection: dict[str, Any] = {
        "schema": "ipfs_accelerate_py/agent-supervisor/duckdb-authority-catalog@2",
        "database": database,
        "schemas": schemas,
        "relations": relations,
        "tables": tables,
        "columns": columns,
        "constraints": constraints,
        "indexes": indexes,
        "views": views,
        "sequences": sequences,
        "types": types,
        "macros": macros,
    }
    projection["catalog_root"] = content_identity(projection)
    return projection


def _read_only_duckdb_catalog(path: Path, *, noun: str) -> dict[str, Any]:
    try:
        import duckdb  # type: ignore
        from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
            connect_duckdb_with_policy,
        )

        connection = connect_duckdb_with_policy(
            duckdb,
            path,
            read_only=True,
            configuration={"threads": 1, "memory_limit": "256MB"},
        )
    except Exception as exc:
        raise MaterializationError(f"{noun} catalog cannot be opened read-only") from exc
    try:
        return _duckdb_catalog_projection(connection)
    except Exception as exc:
        if isinstance(exc, MaterializationError):
            raise
        raise MaterializationError(f"{noun} catalog cannot be reconstructed") from exc
    finally:
        connection.close()


_CONTROL_SEMANTIC_CONTENT_TABLES = frozenset(
    {
        "objectives",
        "objective_revisions",
        "goals",
        "goal_edges",
        "plans",
        "plan_revisions",
        "tasks",
        "task_revisions",
        "task_dependencies",
        "task_outputs",
        "task_acceptance",
        "task_validations",
        "completion_receipts",
        "evidence_nodes",
        "domain_events",
    }
)


def _normalized_control_residual_row(
    table: str,
    row: Mapping[str, Any],
    *,
    migration_receipts: frozenset[str],
) -> dict[str, Any]:
    projected = {str(key): _duckdb_json_value(value) for key, value in row.items()}
    if table == "control_plane_metadata":
        projected["updated_at"] = "<migration-clock>"
    elif table in {"schema_migrations", "schema_migration_attempts"}:
        projected["started_at"] = "<migration-clock>"
        projected["finished_at"] = "<migration-clock>"
        if table == "schema_migration_attempts":
            projected["attempt_id"] = "<migration-attempt-id>"
        else:
            projected["receipt_cid"] = "<migration-receipt-cid>"
        raw_body = row.get("body_json")
        try:
            body = _strict_json_loads(
                str(raw_body), noun=f"{table} migration authority"
            )
        except (TypeError, MaterializationError) as exc:
            raise MaterializationError(
                f"{table} contains malformed migration authority"
            ) from exc
        if not isinstance(body, dict):
            raise MaterializationError(
                f"{table} migration authority must be an object"
            )
        if table == "schema_migrations":
            expected_fields = {
                "schema",
                "version",
                "migration_id",
                "checksum",
                "application_version",
                "tool_version",
                "started_at",
                "finished_at",
                "outcome",
                "schema_fingerprint",
                "error_text",
                "receipt_cid",
            }
            common_fields = expected_fields - {"schema", "error_text", "receipt_cid"}
            claimed_receipt = str(body.get("receipt_cid") or "")
            if (
                set(body) != expected_fields
                or body.get("schema")
                != "ipfs_accelerate_py/agent-supervisor/control-plane-migration-receipt@1"
                or body.get("error_text") != ""
                or claimed_receipt != str(row.get("receipt_cid") or "")
                or claimed_receipt
                != content_identity(
                    {key: value for key, value in body.items() if key != "receipt_cid"}
                )
                or any(body.get(field) != row.get(field) for field in common_fields)
            ):
                raise MaterializationError(
                    "schema_migrations receipt identity differs"
                )
        else:
            expected_fields = {
                "attempt_id",
                "version",
                "migration_id",
                "checksum",
                "application_version",
                "tool_version",
                "started_at",
                "finished_at",
                "outcome",
                "schema_fingerprint",
                "receipt_cid",
            }
            common_fields = expected_fields - {"receipt_cid"}
            if (
                set(body) != expected_fields
                or body.get("attempt_id") != row.get("attempt_id")
                or any(body.get(field) != row.get(field) for field in common_fields)
                or str(body.get("receipt_cid") or "") not in migration_receipts
                or row.get("error_text") != ""
                or row.get("outcome") != "applied"
            ):
                raise MaterializationError(
                    "schema_migration_attempts authority differs"
                )
        normalized_body = dict(body)
        for clock in ("started_at", "finished_at"):
            if clock in normalized_body:
                normalized_body[clock] = "<migration-clock>"
        if table == "schema_migration_attempts":
            normalized_body["attempt_id"] = "<migration-attempt-id>"
        if "receipt_cid" in normalized_body:
            normalized_body["receipt_cid"] = "<migration-receipt-cid>"
        projected["body_json"] = normalized_body
    return projected


def _control_residual_content_projection(
    connection: Any, table_names: set[str]
) -> dict[str, Any]:
    """Bind every non-intent control row not covered by typed projections."""

    column_rows = connection.execute(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = 'main' ORDER BY table_name, ordinal_position"
    ).fetchall()
    columns_by_table: dict[str, list[str]] = {}
    for table, column in column_rows:
        columns_by_table.setdefault(str(table), []).append(str(column))
    tables: dict[str, list[dict[str, Any]]] = {}
    total_rows = 0
    migration_receipts = frozenset(
        str(row[0])
        for row in connection.execute(
            "SELECT receipt_cid FROM schema_migrations ORDER BY receipt_cid"
        ).fetchall()
    )
    for table in sorted(table_names - _CONTROL_SEMANTIC_CONTENT_TABLES):
        columns = columns_by_table.get(table)
        if not columns:
            raise MaterializationError(f"control table {table} has no columns")
        quoted = '"' + table.replace('"', '""') + '"'
        rows = connection.execute(f"SELECT * FROM {quoted}").fetchall()
        total_rows += len(rows)
        if total_rows > 100_000:
            raise MaterializationError("control residual projection exceeds row bound")
        normalized = [
            _normalized_control_residual_row(
                table,
                dict(zip(columns, row, strict=True)),
                migration_receipts=migration_receipts,
            )
            for row in rows
        ]
        normalized.sort(key=_canonical_bytes)
        tables[table] = normalized
    projection: dict[str, Any] = {
        "schema": "ipfs_accelerate_py/agent-supervisor/control-residual-content@1",
        "tables": tables,
        "row_count": total_rows,
    }
    projection["projection_root"] = content_identity(projection)
    return projection


def _read_only_control(
    path: Path,
    population: Mapping[str, Any],
    *,
    expected_stage: str,
) -> dict[str, Any]:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise MaterializationError("DuckDB is unavailable; refusing materialization") from exc
    try:
        from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
            connect_duckdb_with_policy,
        )

        connection = connect_duckdb_with_policy(
            duckdb,
            path,
            read_only=True,
            configuration={"threads": 1, "memory_limit": "256MB"},
        )
    except Exception as exc:
        raise MaterializationError("control store cannot be opened read-only") from exc
    try:
        relation_rows = connection.execute(
            "SELECT table_name, table_type FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
        if any(str(row[1]) not in {"BASE TABLE", "VIEW"} for row in relation_rows):
            raise MaterializationError("control authority relation type differs")
        relation_names = {str(row[0]) for row in relation_rows}
        table_names = {
            str(row[0]) for row in relation_rows if str(row[1]) == "BASE TABLE"
        }
        view_names = {str(row[0]) for row in relation_rows if str(row[1]) == "VIEW"}
        required = {"goals", "goal_edges", "plans", "tasks", "task_dependencies"}
        if not required.issubset(relation_names):
            raise MaterializationError("control store lacks required operational relations")
        forbidden = {
            "semantic_nodes",
            "semantic_edges",
            "semantic_state_roots",
            "proof_obligations",
            "semantic_capsules",
        }
        if relation_names & forbidden:
            raise MaterializationError("control store contains datasets semantic authority")
        table_counts = {
            name: int(
                connection.execute(
                    'SELECT COUNT(*) FROM "' + name.replace('"', '""') + '"'
                ).fetchone()[0]
            )
            for name in sorted(table_names)
        }
        catalog_projection = _duckdb_catalog_projection(connection)
        residual_content_projection = _control_residual_content_projection(
            connection, table_names
        )
        objective_revision_rows = connection.execute(
            "SELECT objective_id, revision, status, body_json "
            "FROM objective_revisions ORDER BY objective_id, revision"
        ).fetchall()
        plan_revision_rows = connection.execute(
            "SELECT plan_cid, revision, body_json "
            "FROM plan_revisions ORDER BY plan_cid, revision"
        ).fetchall()
        objective_revision_projection = [
            {
                "objective_id": str(row[0]),
                "revision": int(row[1]),
                "status": str(row[2]),
                "body": _strict_json_loads(
                    str(row[3]), noun="objective revision body"
                ),
            }
            for row in objective_revision_rows
        ]
        plan_revision_projection = [
            {
                "plan_cid": str(row[0]),
                "revision": int(row[1]),
                "body": _strict_json_loads(
                    str(row[2]), noun="plan revision body"
                ),
            }
            for row in plan_revision_rows
        ]
        task_rows = connection.execute(
            "SELECT task_cid, task_alias, status, body_json, revision "
            "FROM tasks ORDER BY ordinal, task_cid"
        ).fetchall()
        expected_tasks = list(population["tasks"])
        if len(task_rows) != len(expected_tasks):
            raise MaterializationError("control task count differs from population")
        runtime_progress = False
        statuses: dict[str, str] = {}
        statuses_by_cid: dict[str, str] = {}
        task_projection: list[dict[str, Any]] = []
        for row, expected in zip(task_rows, expected_tasks, strict=True):
            task_cid, task_alias, status, body_json = map(str, row[:4])
            revision = int(row[4])
            if task_cid != expected["task_cid"] or task_alias != expected["task_id"]:
                raise MaterializationError("control task identity/order differs")
            try:
                body = _strict_json_loads(
                    body_json, noun=f"{task_alias} task body"
                )
            except MaterializationError as exc:
                raise MaterializationError(f"{task_alias}: task body is invalid JSON") from exc
            for field in (
                "formal_task_content_id",
                "construction_status",
                "completion",
                "review_only",
                "blocked_reason",
                "board_namespace",
            ):
                if body.get(field) != expected.get(field):
                    raise MaterializationError(
                        f"{task_alias}: immutable task body differs at {field}"
                    )
            expected_status = str(expected["status"])
            if expected_stage == "initial":
                if status != expected_status:
                    raise MaterializationError(f"{task_alias}: initial status differs")
            elif expected_status == "blocked":
                if status != "blocked":
                    raise MaterializationError(f"{task_alias}: protected blocker was transitioned")
            elif status != expected_status:
                runtime_progress = True
            statuses[task_alias] = status
            statuses_by_cid[task_cid] = status
            task_projection.append(
                {
                    "task_cid": task_cid,
                    "task_alias": task_alias,
                    "status": status,
                    "revision": revision,
                    "body": body,
                }
            )
        dep_rows = {
            (str(row[0]), str(row[1]))
            for row in connection.execute(
                "SELECT task_cid, dependency_task_cid FROM task_dependencies"
            ).fetchall()
        }
        expected_deps = {
            (str(task["task_cid"]), str(dependency))
            for task in expected_tasks
            for dependency in task["dependencies"]
        }
        if dep_rows != expected_deps:
            raise MaterializationError("control dependency graph differs")
        dependencies_by_task: dict[str, set[str]] = {}
        for task_cid, dependency_cid in dep_rows:
            dependencies_by_task.setdefault(task_cid, set()).add(dependency_cid)
        ready_task_aliases = sorted(
            item["task_alias"]
            for item in task_projection
            if item["status"] == "todo"
            and all(
                statuses_by_cid.get(dependency) == "completed"
                for dependency in dependencies_by_task.get(item["task_cid"], set())
            )
        )
        evidence_rows = connection.execute(
            """
            SELECT evidence_id, parent_evidence_id, task_cid, evidence_kind,
                   digest, body_json
            FROM evidence_nodes
            ORDER BY task_cid, evidence_id
            """
        ).fetchall()
        completion_rows = connection.execute(
            """
            SELECT receipt_cid, task_cid, goal_cid, attempt_id, claim_cid,
                   fencing_token, validation_run_id, evidence_digest, body_json
            FROM completion_receipts
            ORDER BY task_cid, receipt_cid
            """
        ).fetchall()

        def decoded_body(raw: object, *, noun: str) -> dict[str, Any]:
            value = _strict_json_loads(str(raw), noun=f"{noun} body")
            if not isinstance(value, dict):
                raise MaterializationError(f"{noun} body must be an object")
            return value

        evidence_projection = [
            {
                "evidence_id": str(row[0]),
                "parent_evidence_id": str(row[1] or ""),
                "task_cid": str(row[2]),
                "evidence_kind": str(row[3]),
                "digest": str(row[4]),
                "body": decoded_body(row[5], noun="control evidence"),
            }
            for row in evidence_rows
        ]
        completion_projection = [
            {
                "receipt_cid": str(row[0]),
                "task_cid": str(row[1]),
                "goal_cid": str(row[2]),
                "attempt_id": str(row[3] or ""),
                "claim_cid": str(row[4] or ""),
                "fencing_token": int(row[5]),
                "validation_run_id": str(row[6] or ""),
                "evidence_digest": str(row[7]),
                "body": decoded_body(row[8], noun="control completion receipt"),
            }
            for row in completion_rows
        ]
        goal_cids = {
            str(row[0]) for row in connection.execute("SELECT goal_cid FROM goals").fetchall()
        }
        if goal_cids != {str(item["goal_cid"]) for item in population["objectives"]}:
            raise MaterializationError("control goal identities differ")
        plan_rows = connection.execute("SELECT plan_cid FROM plans").fetchall()
        if [str(row[0]) for row in plan_rows] != [str(population["plan_root_cid"])]:
            raise MaterializationError("control plan identity differs")
        with tempfile.TemporaryDirectory(prefix="lgcvf-control-projection-") as temporary:
            from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
                verify_datasets_authoritative_operational_schema,
            )
            from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
                DatabaseTaskSource,
            )
            from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
                IntentRepositoryError,
            )

            copied = Path(temporary) / "control.duckdb"
            shutil.copyfile(path, copied)
            schema_verification = _plain_json(
                verify_datasets_authoritative_operational_schema(copied)
            )
            schema_verification.pop("database_path", None)
            projection_source = DatabaseTaskSource(copied, install_schema=False)
            try:
                plan_projection = _plain_json(projection_source.plan_projection())
                revision_histories = [
                    _plain_json(
                        projection_source.task_revision_history_projection(
                            str(item["task_cid"])
                        )
                    )
                    for item in population["tasks"]
                ]
                completion_evidence_before = _plain_json(
                    projection_source.completion_evidence_projection()
                )
                events: list[dict[str, Any]] = []
                after_sequence = 0
                while True:
                    page = projection_source.intent.list_events(
                        after_global_sequence=after_sequence,
                        limit=1_000,
                    )
                    if not page:
                        break
                    for raw_event in page:
                        event = _plain_json(raw_event)
                        body = event.get("body")
                        if not isinstance(body, Mapping):
                            raise MaterializationError("control event body is malformed")
                        expected_event_id = content_identity(
                            {
                                "stream_id": event.get("stream_id"),
                                "sequence": event.get("sequence"),
                                "global_sequence": event.get("global_sequence"),
                                "event_type": event.get("event_type"),
                                "body": body,
                            }
                        )
                        if event.get("event_id") != expected_event_id:
                            raise MaterializationError(
                                "control event content identity differs"
                            )
                        events.append(event)
                    next_sequence = int(page[-1]["global_sequence"])
                    if next_sequence <= after_sequence:
                        raise MaterializationError("control event cursor did not advance")
                    after_sequence = next_sequence
                    if len(page) < 1_000:
                        break
                projection_source.intent.rebuild_projections_from_events()
                rebuilt_plan = _plain_json(projection_source.plan_projection())
                rebuilt_histories = [
                    _plain_json(
                        projection_source.task_revision_history_projection(
                            str(item["task_cid"])
                        )
                    )
                    for item in population["tasks"]
                ]
                rebuilt_completion = _plain_json(
                    projection_source.completion_evidence_projection()
                )
                if rebuilt_plan != plan_projection:
                    raise MaterializationError(
                        "control event replay does not reconstruct plan authority: "
                        f"{content_identity(plan_projection)} != "
                        f"{content_identity(rebuilt_plan)}"
                    )
                if rebuilt_histories != revision_histories:
                    raise MaterializationError(
                        "control event replay does not reconstruct task revisions"
                    )
                if rebuilt_completion != completion_evidence_before:
                    raise MaterializationError(
                        "control event replay does not reconstruct completion evidence"
                    )
                def semantic_event_body(event: Mapping[str, Any]) -> dict[str, Any]:
                    """Normalize only the two repository-issued event clocks."""

                    wrapper = event.get("body")
                    if not isinstance(wrapper, Mapping) or set(wrapper) != {
                        "schema",
                        "event_type",
                        "subject_id",
                        "body",
                        "recorded_at",
                        "owner_id",
                    }:
                        raise MaterializationError(
                            "control event wrapper fields differ"
                        )
                    if (
                        wrapper.get("schema")
                        != "ipfs_accelerate_py/agent-supervisor/intent-event@1"
                        or wrapper.get("event_type") != event.get("event_type")
                        or wrapper.get("recorded_at") != event.get("recorded_at")
                        or not str(wrapper.get("recorded_at") or "")
                    ):
                        raise MaterializationError(
                            "control event wrapper identity differs"
                        )
                    payload = wrapper.get("body")
                    if not isinstance(payload, Mapping):
                        raise MaterializationError(
                            "control event payload is malformed"
                        )
                    normalized_payload = dict(payload)
                    event_type = str(event.get("event_type") or "")
                    recorded_event_types = {
                        "intent.objective_upserted",
                        "intent.goal_upserted",
                        "intent.plan_upserted",
                        "intent.task_upserted",
                        "intent.completion_recorded",
                    }
                    if event_type in recorded_event_types:
                        if not str(normalized_payload.get("recorded_at") or ""):
                            raise MaterializationError(
                                "control recovery event clock differs"
                            )
                        normalized_payload.pop("recorded_at")
                    elif event_type == "intent.evidence_recorded":
                        created_at = normalized_payload.pop("created_at", None)
                        if not str(created_at or ""):
                            raise MaterializationError(
                                "control evidence event clock is malformed"
                            )
                    elif event_type != "intent.goal_edge_linked":
                        raise MaterializationError(
                            "control recovery event type differs"
                        )
                    normalized_wrapper = dict(wrapper)
                    normalized_wrapper.pop("recorded_at")
                    normalized_wrapper["body"] = normalized_payload
                    return _plain_json(normalized_wrapper)

                semantic_events = [
                    {
                        "stream_id": item["stream_id"],
                        "sequence": item["sequence"],
                        "global_sequence": item["global_sequence"],
                        "event_type": item["event_type"],
                        "task_cid": item["task_cid"],
                        "attempt_id": item["attempt_id"],
                        "session_id": item["session_id"],
                        "body": semantic_event_body(item),
                    }
                    for item in events
                ]
            except IntentRepositoryError as exc:
                raise MaterializationError(
                    "control typed authority contains invalid JSON"
                ) from exc
            finally:
                projection_source.close()
        return {
            "task_count": len(task_rows),
            "goal_count": len(goal_cids),
            "dependency_count": len(dep_rows),
            "statuses": statuses,
            "tasks": task_projection,
            "ready_task_aliases": ready_task_aliases,
            "evidence": evidence_projection,
            "completion_receipts": completion_projection,
            "plan_projection": plan_projection,
            "task_revision_histories": revision_histories,
            "objective_revision_history": objective_revision_projection,
            "plan_revision_history": plan_revision_projection,
            "schema_verification": schema_verification,
            "catalog_projection": catalog_projection,
            "residual_content_projection": residual_content_projection,
            "table_counts": table_counts,
            "event_stream_root": content_identity(events),
            "semantic_event_stream_root": content_identity(semantic_events),
            "semantic_events": semantic_events,
            "runtime_progress_observed": runtime_progress,
            "relation_count": len(relation_names),
            "relation_inventory": {
                "tables": sorted(table_names),
                "views": sorted(view_names),
            },
        }
    finally:
        connection.close()


def _read_only_execution(path: Path, *, expected_stage: str) -> dict[str, Any]:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise MaterializationError("DuckDB is unavailable") from exc
    try:
        from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
            connect_duckdb_with_policy,
        )

        connection = connect_duckdb_with_policy(
            duckdb,
            path,
            read_only=True,
            configuration={"threads": 1, "memory_limit": "256MB"},
        )
    except Exception as exc:
        raise MaterializationError("execution store cannot be opened read-only") from exc
    activity_tables = (
        "database_task_attempts",
        "attempt_phases",
        "provider_invocations",
        "effect_claims",
        "daemon_execution_events",
    )
    try:
        table_rows = connection.execute(
            "SELECT table_name, table_type FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
        if any(str(row[1]) != "BASE TABLE" for row in table_rows):
            raise MaterializationError("execution authority contains a non-table relation")
        observed_tables = {str(row[0]) for row in table_rows}
        expected_tables = {"daemon_execution_metadata", *activity_tables}
        if observed_tables != expected_tables:
            raise MaterializationError("execution store table inventory differs")
        catalog_projection = _duckdb_catalog_projection(connection)
        column_rows = connection.execute(
            """
            SELECT table_name, column_name, data_type, is_nullable,
                   COALESCE(column_default, ''), ordinal_position
            FROM information_schema.columns
            WHERE table_schema = 'main'
            ORDER BY table_name, ordinal_position
            """
        ).fetchall()
        index_rows = connection.execute(
            "SELECT index_name, table_name, sql FROM duckdb_indexes() "
            "WHERE schema_name = 'main' ORDER BY index_name"
        ).fetchall()
        schema_inventory = {
            "tables": sorted(observed_tables),
            "columns": [
                {
                    "table": str(row[0]),
                    "column": str(row[1]),
                    "type": str(row[2]),
                    "nullable": str(row[3]),
                    "default": str(row[4] or ""),
                    "ordinal": int(row[5]),
                }
                for row in column_rows
            ],
            "indexes": [
                {
                    "index": str(row[0]),
                    "table": str(row[1]),
                    "sql": str(row[2]),
                }
                for row in index_rows
            ],
        }
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in activity_tables
        }
        metadata = {
            str(row[0]): str(row[1])
            for row in connection.execute(
                "SELECT key, value FROM daemon_execution_metadata ORDER BY key"
            ).fetchall()
        }
    except Exception as exc:
        if isinstance(exc, MaterializationError):
            raise
        raise MaterializationError("execution store lacks the daemon schema") from exc
    finally:
        connection.close()
    if expected_stage == "initial" and any(counts.values()):
        raise MaterializationError("initial execution store already contains attempts/effects")
    expected_metadata_fields = {
        "interface",
        "schema",
        "authority_mode",
        "logical_owner_session_id",
        "process_instance_id",
        "state_schema_revision",
        "control_schema_profile_id",
        "control_schema_fingerprint",
    }
    if set(metadata) != expected_metadata_fields or any(not value for value in metadata.values()):
        raise MaterializationError("execution store metadata differs")
    if (
        metadata["interface"] != "DatabaseImplementationDaemon@1"
        or metadata["schema"]
        != "ipfs_accelerate_py/agent-supervisor/database-implementation-daemon@1"
        or metadata["authority_mode"] != "embedded"
        or metadata["state_schema_revision"] != EXPECTED_SCHEMA_REVISION
    ):
        raise MaterializationError("execution store authority metadata differs")
    return {
        "row_counts": counts,
        "metadata": metadata,
        "schema_inventory": schema_inventory,
        "catalog_projection": catalog_projection,
        "runtime_progress_observed": any(counts.values()),
    }


def _verify_read_only_core(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    root: Path = ROOT,
    expected_stage: str = "live",
) -> dict[str, Any]:
    """Reconstruct immutable authority without creating locks or writing stores."""

    if expected_stage not in {"initial", "live"}:
        raise MaterializationError("expected_stage must be 'initial' or 'live'")
    paths = _paths(config, root=root)
    stores = {key: path for key, path in paths.items() if key != "receipt"}
    missing = [key for key, path in stores.items() if not path.is_file()]
    if missing:
        raise MaterializationError(f"operational stores are absent: {', '.join(missing)}")
    before = {
        key: (path.stat().st_size, path.stat().st_mtime_ns, _sha256_file(path))
        for key, path in stores.items()
    }
    control = _read_only_control(paths["control"], population, expected_stage=expected_stage)
    try:
        coordination = read_coordination_registry_projection(paths["coordination"])
    except Exception as exc:
        raise MaterializationError("coordination registry fails read-only verification") from exc
    expected_registry = {
        str(item["task_cid"]): str(item["task_id"]) for item in population["tasks"]
    }
    registered = {
        str(item["task_cid"]): str(item["task_id"]) for item in coordination.get("tasks", ())
    }
    # The typed coordination projection is deliberately ordered by durable
    # CID, while the formal population is ordered by task alias/ordinal.
    if registered != expected_registry:
        raise MaterializationError("coordination task identities differ")
    expected_edges = {
        (str(task["task_cid"]), str(dep))
        for task in population["tasks"]
        for dep in task["dependencies"]
    }
    coordination_edges = {
        (str(item["task_cid"]), str(item["dependency_task_cid"]))
        for item in coordination.get("dependency_edges", ())
    }
    if coordination_edges != expected_edges:
        raise MaterializationError("coordination dependency graph differs")
    coordination_catalog = _read_only_duckdb_catalog(
        paths["coordination"], noun="coordination"
    )
    execution = _read_only_execution(paths["execution"], expected_stage=expected_stage)
    after = {
        key: (path.stat().st_size, path.stat().st_mtime_ns, _sha256_file(path))
        for key, path in stores.items()
    }
    if before != after:
        raise MaterializationError("read-only verification changed an operational store")
    report = {
        "schema": VERIFICATION_SCHEMA,
        "valid": True,
        "verification_mode": "read_only",
        "expected_stage": expected_stage,
        "population_root": population["population_root"],
        "plan_root_cid": population["plan_root_cid"],
        "repository_tree_id": population["repository_tree_id"],
        "control": control,
        "coordination": {
            "counts": coordination["counts"],
            "projection_root": coordination["projection_root"],
            "catalog_projection": coordination_catalog,
        },
        "execution": execution,
        "stores_unchanged": True,
        "maximum_writer_processes": 1,
        "quack_qualified": False,
    }
    report["verification_root"] = content_identity(report)
    return report


def verify_read_only(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    root: Path = ROOT,
    expected_stage: str = "live",
) -> dict[str, Any]:
    """Verify ordinary control planes; fresh recovery requires strict replay."""

    if _targets_fresh_recovery_generation(config, root=root):
        raise MaterializationError(
            "fresh recovery targets require verify-fresh-recovery; generic "
            "verification is not admission authority"
        )
    return _verify_read_only_core(
        config, population, root=root, expected_stage=expected_stage
    )


def _load_recovery_json(path: Path, *, noun: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise MaterializationError(f"{noun} is absent or unreadable") from exc
    value = _strict_json_loads(raw, noun=noun)
    if not isinstance(value, dict):
        raise MaterializationError(f"{noun} must be an object")
    if raw != _canonical_bytes(value) + b"\n":
        raise MaterializationError(f"{noun} bytes are not canonical")
    return value


def _fresh_recovery_paths(
    config: Mapping[str, Any], *, root: Path
) -> dict[str, Path]:
    policy = _fresh_recovery_policy(config)
    runtime = config.get("runtime_paths")
    if not isinstance(runtime, Mapping):
        raise MaterializationError("runtime_paths is required")
    target = _lexical_safe_path(
        root,
        policy.get("target_runtime_root"),
        field="fresh_generation_recovery.target_runtime_root",
    )
    evidence = _safe_path(root, runtime.get("evidence"), field="runtime_paths.evidence")
    recovery = evidence / "fresh-generation-recovery"
    return {
        **_paths(config, root=root),
        "target": target,
        "recovery": recovery,
        "recovery_receipt": recovery / "recovery-receipt.json",
    }


def _fresh_recovery_target_state(
    config: Mapping[str, Any], *, root: Path
) -> tuple[Path, str]:
    """Inspect the unresolved target path and reject every link/special parent."""

    policy = _fresh_recovery_policy(config)
    target = _lexical_safe_path(
        root,
        policy.get("target_runtime_root"),
        field="fresh_generation_recovery.target_runtime_root",
    )
    resolved_root = root.resolve(strict=True)
    relative = target.relative_to(resolved_root)
    cursor = resolved_root
    for index, component in enumerate(relative.parts):
        cursor = cursor / component
        try:
            status = cursor.lstat()
        except FileNotFoundError:
            return target, "absent"
        except OSError as exc:
            raise MaterializationError("fresh recovery target cannot be inspected") from exc
        if stat.S_ISLNK(status.st_mode):
            raise MaterializationError("fresh recovery target path contains a symlink")
        final = index == len(relative.parts) - 1
        if final:
            if not stat.S_ISDIR(status.st_mode):
                raise MaterializationError("fresh recovery target has wrong file type")
            return target, "present"
        if not stat.S_ISDIR(status.st_mode):
            raise MaterializationError("fresh recovery target parent has wrong file type")
    return target, "present"


def _validate_fresh_recovery_layout(
    config: Mapping[str, Any],
    *,
    root: Path,
    manifest_path: Path | None = None,
) -> None:
    """Reject links/special files and authorities outside the atomic target."""

    resolved_root = root.resolve(strict=True)
    policy = _fresh_recovery_policy(config)
    program = config.get("database_program")
    runtime = config.get("runtime_paths")
    if not isinstance(program, Mapping) or not isinstance(runtime, Mapping):
        raise MaterializationError("fresh recovery layout configuration is absent")
    target = resolved_root / Path(str(policy["target_runtime_root"]))
    control = resolved_root / Path(str(program["store_id"]))
    evidence = resolved_root / Path(str(runtime["evidence"]))
    operational_directories = {
        field: resolved_root / Path(str(runtime[field]))
        for field in ("state", "worktrees", "merge_queue", "logs")
    }
    operational_directories.update(
        {
            "event_store_path": resolved_root
            / Path(str(program["event_store_path"])),
            "runtime_registry_path": resolved_root
            / Path(str(program["runtime_registry_path"])),
            "worktree_root": resolved_root / Path(str(program["worktree_root"])),
        }
    )
    coordination = control.with_name(f"{control.stem}.coordination.duckdb")
    execution = control.with_name(f"{control.stem}.execution.duckdb")
    lock_files = {
        "control lock": control.parent / f".{control.name}.lock",
        "control intent lock": control.parent / f".{control.name}.intent.lock",
        "control migration lock": control.parent / f".{control.name}.migration.lock",
        "coordination lock": coordination.parent / f".{coordination.name}.lock",
        "execution lock": execution.parent / f".{execution.name}.lock",
    }
    files = {
        "control": control,
        "coordination": coordination,
        "execution": execution,
        "bootstrap receipt": evidence / "bootstrap" / "materialization.json",
        "recovery receipt": evidence
        / "fresh-generation-recovery"
        / "recovery-receipt.json",
        **lock_files,
    }
    if manifest_path is not None:
        files["recovery manifest"] = manifest_path
    directories = {
        "target": target,
        "control parent": control.parent,
        "evidence": evidence,
        "bootstrap": evidence / "bootstrap",
        "recovery": evidence / "fresh-generation-recovery",
    }

    def checked_components(path: Path, *, final_directory: bool, noun: str) -> None:
        try:
            relative = path.relative_to(resolved_root)
        except ValueError as exc:
            raise MaterializationError(f"fresh recovery {noun} escapes root") from exc
        cursor = resolved_root
        for index, component in enumerate(relative.parts):
            cursor = cursor / component
            try:
                status = cursor.lstat()
            except OSError as exc:
                raise MaterializationError(f"fresh recovery {noun} is absent") from exc
            if stat.S_ISLNK(status.st_mode):
                raise MaterializationError(f"fresh recovery {noun} contains a symlink")
            final = index == len(relative.parts) - 1
            if final and final_directory:
                valid = stat.S_ISDIR(status.st_mode)
            elif final:
                valid = stat.S_ISREG(status.st_mode)
            else:
                valid = stat.S_ISDIR(status.st_mode)
            if not valid:
                raise MaterializationError(f"fresh recovery {noun} has wrong file type")
            mode = stat.S_IMODE(status.st_mode)
            inside_target = cursor == target or target in cursor.parents
            if inside_target and (
                status.st_uid != os.geteuid() or mode & 0o022
            ):
                raise MaterializationError(
                    f"fresh recovery {noun} permissions differ"
                )
            if inside_target and stat.S_ISDIR(status.st_mode) and mode & 0o700 != 0o700:
                raise MaterializationError(
                    f"fresh recovery {noun} directory is not owner-accessible"
                )
            if (
                final
                and not final_directory
                and (status.st_nlink != 1 or status.st_uid != os.geteuid())
            ):
                raise MaterializationError(f"fresh recovery {noun} file identity differs")
            if final and not final_directory:
                required_owner_mode = (
                    0o600
                    if noun in {"control", "coordination", "execution"}
                    or noun.endswith(" lock")
                    else 0o400
                )
                if mode & required_owner_mode != required_owner_mode:
                    raise MaterializationError(
                        f"fresh recovery {noun} file is not owner-accessible"
                    )
        try:
            path.resolve(strict=True).relative_to(target.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise MaterializationError(f"fresh recovery {noun} escapes target") from exc

    for noun, directory in directories.items():
        checked_components(directory, final_directory=True, noun=noun)
    for noun, path in files.items():
        checked_components(path, final_directory=False, noun=noun)
    for noun, directory in operational_directories.items():
        if directory.is_symlink():
            raise MaterializationError(
                f"fresh recovery {noun} directory contains a symlink"
            )
        if directory.exists():
            checked_components(directory, final_directory=True, noun=noun)
            if any(directory.iterdir()):
                raise MaterializationError(
                    f"fresh recovery {noun} directory is not empty"
                )
    revision_store = operational_directories["state"] / "plan-revision-store"
    if revision_store.exists() or revision_store.is_symlink():
        raise MaterializationError(
            "fresh recovery operational plan-revision-store must be absent"
        )
    required_root_entries = {
        control.name,
        coordination.name,
        execution.name,
        evidence.name,
        *(path.name for path in lock_files.values()),
    }
    allowed_root_entries = required_root_entries | {
        path.relative_to(target).parts[0]
        for path in operational_directories.values()
        if path != target
    }
    observed_root_entries = {item.name for item in target.iterdir()}
    if not required_root_entries.issubset(observed_root_entries) or not (
        observed_root_entries <= allowed_root_entries
    ):
        raise MaterializationError(
            "fresh recovery target inventory differs: "
            f"observed={sorted(observed_root_entries)!r}; "
            f"required={sorted(required_root_entries)!r}; "
            f"allowed={sorted(allowed_root_entries)!r}"
        )
    if {item.name for item in evidence.iterdir()} != {
        "bootstrap",
        "fresh-generation-recovery",
    }:
        raise MaterializationError("fresh recovery evidence inventory differs")
    bootstrap = evidence / "bootstrap"
    if {item.name for item in bootstrap.iterdir()} != {"materialization.json"}:
        raise MaterializationError("fresh recovery bootstrap inventory differs")
    recovery = evidence / "fresh-generation-recovery"
    expected_recovery_entries = {"recovery-receipt.json"}
    if manifest_path is not None:
        expected_recovery_entries.add(manifest_path.name)
    else:
        manifests = {item.name for item in recovery.glob("*.manifest.json")}
        if len(manifests) != 1:
            raise MaterializationError("fresh recovery manifest inventory differs")
        expected_recovery_entries.update(manifests)
    if {item.name for item in recovery.iterdir()} != expected_recovery_entries:
        raise MaterializationError("fresh recovery artifact inventory differs")


def _validation_spec(task: Mapping[str, Any]) -> dict[str, Any]:
    task_id = str(task.get("task_id") or "")
    validations = task.get("validations")
    if not isinstance(validations, list) or len(validations) != 1:
        raise MaterializationError(f"{task_id}: exactly one declared validation is required")
    declared = str(validations[0] or "")
    try:
        argv = shlex.split(declared)
    except ValueError as exc:
        raise MaterializationError(f"{task_id}: declared validation is malformed") from exc
    if (
        len(argv) != 5
        or argv[:4] != ["python", "-m", "pytest", "-q"]
        or Path(argv[4]).is_absolute()
        or ".." in Path(argv[4]).parts
        or not argv[4].endswith(".py")
    ):
        raise MaterializationError(
            f"{task_id}: recovery permits only one bounded local pytest target"
        )
    owner = str(task.get("owning_repository") or "")
    if owner == "ipfs_datasets_py":
        working_directory = "ipfs_datasets_py"
        python_path = ".:.."
        historical_command = (
            "cd ipfs_datasets_py && export PYTHONPATH=.:.. && " + declared
        )
    elif owner == "ipfs_accelerate_py":
        working_directory = "."
        python_path = "ipfs_datasets_py"
        historical_command = "export PYTHONPATH=ipfs_datasets_py && " + declared
    else:
        raise MaterializationError(f"{task_id}: recovery repository owner is unsupported")
    spec = {
        "task_id": task_id,
        "task_cid": str(task.get("task_cid") or ""),
        "declared_command": declared,
        "historical_command": historical_command,
        "argv": argv,
        "working_directory": working_directory,
        "python_path": python_path,
        "timeout_seconds": 900,
        "provider_route": "none",
        "network_client_required": False,
    }
    spec["validation_spec_cid"] = content_identity(spec)
    return spec


def _gitlink_at(repository: Path, commit: str, path: str) -> str:
    fields = _git(repository, "ls-tree", commit, "--", path).split()
    if len(fields) < 3 or fields[0:2] != ["160000", "commit"]:
        raise MaterializationError(f"{commit}: {path} is not a gitlink")
    return fields[2]


def _changed_paths_for_candidate(
    *,
    source_root: Path,
    task: Mapping[str, Any],
    completed: Mapping[str, Any],
    candidate_commit: str,
) -> list[str]:
    task_id = str(task.get("task_id") or "")
    metadata = completed.get("metadata")
    if not isinstance(metadata, Mapping):
        raise MaterializationError(f"{task_id}: completed record metadata is absent")
    baseline = str(metadata.get("baseline_ref") or "")
    parents = _git(source_root, "show", "-s", "--format=%P", candidate_commit).split()
    if len(parents) != 1 or parents[0] != baseline:
        raise MaterializationError(f"{task_id}: candidate baseline/parent differs")
    candidate_tree = _git(source_root, "show", "-s", "--format=%T", candidate_commit)
    if candidate_tree != str(metadata.get("candidate_tree") or ""):
        raise MaterializationError(f"{task_id}: candidate tree differs")
    expected_outputs = {
        str(item.get("path") or "")
        for item in task.get("outputs") or ()
        if isinstance(item, Mapping)
    }
    if not expected_outputs:
        raise MaterializationError(f"{task_id}: canonical output boundary is empty")
    owner = str(task.get("owning_repository") or "")
    top_level = set(
        filter(
            None,
            _git(
                source_root,
                "diff",
                "--name-only",
                "--no-renames",
                baseline,
                candidate_commit,
                "--",
            ).splitlines(),
        )
    )
    if owner == "ipfs_datasets_py":
        if top_level != {"ipfs_datasets_py"}:
            raise MaterializationError(f"{task_id}: candidate changed outside its gitlink")
        before = _gitlink_at(source_root, baseline, "ipfs_datasets_py")
        after = _gitlink_at(source_root, candidate_commit, "ipfs_datasets_py")
        datasets = source_root / "ipfs_datasets_py"
        nested = set(
            filter(
                None,
                _git(
                    datasets,
                    "diff",
                    "--name-only",
                    "--no-renames",
                    before,
                    after,
                    "--",
                ).splitlines(),
            )
        )
        if not nested or not nested.issubset(expected_outputs):
            raise MaterializationError(f"{task_id}: nested changes exceed declared outputs")
        _git(datasets, "merge-base", "--is-ancestor", after, "HEAD")
        changed = sorted(nested)
    else:
        if not top_level or not top_level.issubset(expected_outputs):
            raise MaterializationError(f"{task_id}: candidate changes exceed declared outputs")
        changed = sorted(top_level)
    historical_task = metadata.get("task")
    historical_outputs = (
        historical_task.get("outputs")
        if isinstance(historical_task, Mapping)
        else None
    )
    expected_historical_outputs = sorted(
        ("ipfs_datasets_py/" + path if owner == "ipfs_datasets_py" else path)
        for path in expected_outputs
    )
    if not isinstance(historical_outputs, list) or sorted(
        map(str, historical_outputs)
    ) != expected_historical_outputs:
        raise MaterializationError(f"{task_id}: historical output boundary differs")
    return changed


def _validate_historical_validation(
    *,
    task: Mapping[str, Any],
    completed: Mapping[str, Any],
    candidate_commit: str,
    candidate_tree: str,
) -> dict[str, Any]:
    task_id = str(task.get("task_id") or "")
    metadata = completed.get("metadata")
    proof = metadata.get("validation_proof") if isinstance(metadata, Mapping) else None
    spec = _validation_spec(task)
    if (
        not isinstance(proof, Mapping)
        or proof.get("attempted") is not True
        or proof.get("passed") is not True
        or proof.get("returncode") != 0
        or proof.get("target_commit") != candidate_commit
        or proof.get("target_tree") != candidate_tree
        or proof.get("repository_tree_id") != "git-tree:" + candidate_tree
        or not isinstance(metadata, Mapping)
        or metadata.get("implementation_commit") != candidate_commit
        or metadata.get("candidate_tree") != candidate_tree
        or metadata.get("repository_tree_id") != "git-tree:" + candidate_tree
        or proof.get("cache_hits") != 0
        or proof.get("cache_misses") != 1
        or proof.get("fallbacks") not in ([], ())
    ):
        raise MaterializationError(f"{task_id}: historical validation did not pass exactly")
    results = proof.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise MaterializationError(f"{task_id}: historical validation result count differs")
    result = results[0]
    if not isinstance(result, Mapping):
        raise MaterializationError(f"{task_id}: historical validation result is malformed")
    selection = proof.get("selection")
    decisions = selection.get("decisions") if isinstance(selection, Mapping) else None
    stages = proof.get("stages")
    digest = str(result.get("validation_result_digest") or "")
    validation_id = str(result.get("validation_id") or "")
    if (
        result.get("command") != spec["historical_command"]
        or result.get("returncode") != 0
        or result.get("cache_hit") is not False
        or result.get("timed_out") is not False
        or result.get("stage") != "targeted"
        or not validation_id.startswith("declared:")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or not isinstance(selection, Mapping)
        or selection.get("selected_count") != 1
        or selection.get("skipped_count") != 0
        or selection.get("fallback_count") != 0
        or selection.get("unresolved_fallback_count") != 0
        or selection.get("escalated") is not False
        or not isinstance(decisions, list)
        or len(decisions) != 1
        or not isinstance(decisions[0], Mapping)
        or decisions[0].get("command") != spec["historical_command"]
        or decisions[0].get("validation_id") != validation_id
        or decisions[0].get("source") != "declared"
        or decisions[0].get("selected") is not True
        or decisions[0].get("fallback") is not False
        or not isinstance(stages, list)
        or len(stages) != 1
        or stages[0].get("stage") != "targeted"
        or stages[0].get("planned_count") != 1
        or stages[0].get("executed_count") != 1
        or stages[0].get("passed") is not True
    ):
        raise MaterializationError(
            f"{task_id}: historical evidence is not one uncached declared validation"
        )
    return {
        "validation_spec": spec,
        "historical_validation_id": validation_id,
        "historical_validation_result_digest": digest,
    }


def _validate_retained_completion_binding(
    *,
    config: Mapping[str, Any],
    policy: Mapping[str, Any],
    population: Mapping[str, Any],
    source_root: Path,
) -> dict[str, Any]:
    path, receipt_bytes, receipt_sha256 = _read_regular_evidence_bytes(
        source_root,
        policy.get("retained_revision_receipt_path"),
        field="fresh_generation_recovery.retained_revision_receipt_path",
    )
    if receipt_sha256 != policy.get("retained_revision_receipt_sha256"):
        raise MaterializationError("retained revision receipt digest differs")
    receipt = _decode_evidence_json(receipt_bytes, noun="retained revision receipt")
    if receipt.get("schema") != SUCCESSOR_RECEIPT_SCHEMA:
        raise MaterializationError("retained revision receipt schema differs")
    receipt_claimed = str(receipt.pop("receipt_cid", ""))
    receipt_observed = content_identity(receipt)
    receipt["receipt_cid"] = receipt_claimed
    if not receipt_claimed or receipt_claimed != receipt_observed:
        raise MaterializationError("retained revision receipt identity differs")
    if set(receipt) != {
        "schema",
        "receipt_cid",
        "authority_mode",
        "predecessor_plan_cid",
        "candidate_plan_cid",
        "predecessor_revision_cid",
        "successor_revision_cid",
        "delta_cid",
        "preview_cid",
        "recovery_manifest_cid",
        "predecessor_evidence_root",
        "predecessor_archive_sha256",
        "candidate_task_spec_cids",
        "retained_task_cids",
        "added_task_cids",
        "completed_task_cids",
        "blocked_task_cids",
        "retained_completion_binding",
        "protected_blocker_binding",
        "bootstrap_receipt_cid",
        "bootstrap_receipt_sha256",
        "database_sha256_before",
        "database_sha256_after",
        "post_composite_projection",
        "plan_revision_apply_receipt",
        "historical_status_rewritten",
        "manual_or_external_task_completed",
        "production_authorized",
        "bootstrap_receipt_mutated",
        "execution_store_mutated",
    }:
        raise MaterializationError("retained revision receipt fields differ")
    if receipt.get("receipt_cid") != policy.get("retained_revision_receipt_cid"):
        raise MaterializationError("retained revision receipt CID differs")
    plan_binding = population.get("plans")
    if not isinstance(plan_binding, list) or len(plan_binding) != 1:
        raise MaterializationError("retained revision population plan is ambiguous")
    plan_root = str(population.get("plan_root_cid") or "")
    predecessor_plan = str(plan_binding[0].get("predecessor_plan_cid") or "")
    successor_revision = str(policy.get("retained_successor_revision_cid") or "")
    delta_cid = str(policy.get("retained_delta_cid") or "")
    if (
        path.stem != successor_revision
        or receipt.get("successor_revision_cid") != successor_revision
        or receipt.get("candidate_plan_cid") != plan_root
        or receipt.get("predecessor_plan_cid") != predecessor_plan
        or receipt.get("delta_cid") != delta_cid
        or any(
            receipt.get(field) is not False
            for field in (
                "historical_status_rewritten",
                "manual_or_external_task_completed",
                "production_authorized",
                "bootstrap_receipt_mutated",
                "execution_store_mutated",
            )
        )
    ):
        raise MaterializationError("retained revision authority binding differs")
    apply_receipt = receipt.get("plan_revision_apply_receipt")
    if (
        not isinstance(apply_receipt, Mapping)
        or apply_receipt.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/plan-revision-apply-receipt@1"
        or apply_receipt.get("committed") is not True
        or apply_receipt.get("state") != "committed"
        or apply_receipt.get("quarantined") is not False
        or apply_receipt.get("reason_codes") not in ([], ())
        or apply_receipt.get("revision_cid") != successor_revision
        or apply_receipt.get("plan_root_cid") != plan_root
        or apply_receipt.get("delta_cid") != delta_cid
    ):
        raise MaterializationError("retained revision apply receipt differs")
    binding = receipt.get("retained_completion_binding")
    if not isinstance(binding, dict):
        raise MaterializationError("retained completion binding is absent")
    claimed = str(binding.get("binding_cid") or "")
    observed = content_identity(
        {key: value for key, value in binding.items() if key != "binding_cid"}
    )
    if claimed != policy.get("retained_completion_binding_cid") or claimed != observed:
        raise MaterializationError("retained completion binding identity differs")
    task_cids = {
        str(item.get("task_id") or ""): str(item.get("task_cid") or "")
        for item in population.get("tasks") or ()
        if isinstance(item, Mapping)
    }
    predecessor, predecessor_archive_sha256 = _load_predecessor_plan_evidence(
        config, root=source_root
    )
    added_cid = task_cids.get("LGCVF-113", "")
    expected_retained_cids = {task.content_id for task in predecessor.tasks}
    expected_ids = (
        FRESH_RECOVERY_CONSTRUCTION_COMPLETIONS
        + FRESH_RECOVERY_MERGE_COMPLETIONS[:-1]
    )
    expected_cids = {task_cids[item] for item in expected_ids}
    expected_blocked_cids = {
        task_cids[item]
        for item in ("LGCVF-080", *FRESH_RECOVERY_PROTECTED_BLOCKERS)
    }
    if (
        predecessor.content_id != predecessor_plan
        or predecessor_archive_sha256 != receipt.get("predecessor_archive_sha256")
        or set(map(str, receipt.get("retained_task_cids") or ()))
        != expected_retained_cids
        or set(map(str, receipt.get("added_task_cids") or ())) != {added_cid}
        or set(map(str, receipt.get("completed_task_cids") or ())) != expected_cids
        or set(map(str, receipt.get("blocked_task_cids") or ()))
        != expected_blocked_cids
    ):
        raise MaterializationError("retained revision task-set binding differs")
    blocker = receipt.get("protected_blocker_binding")
    blocker_tasks = blocker.get("tasks") if isinstance(blocker, Mapping) else None
    blocker_claimed = (
        str(blocker.get("binding_cid") or "") if isinstance(blocker, Mapping) else ""
    )
    blocker_observed = (
        content_identity(
            {key: value for key, value in blocker.items() if key != "binding_cid"}
        )
        if isinstance(blocker, Mapping)
        else ""
    )
    if (
        blocker_claimed != policy.get("retained_protected_blocker_binding_cid")
        or blocker_claimed != blocker_observed
        or not isinstance(blocker_tasks, Mapping)
        or set(blocker_tasks) != set(FRESH_RECOVERY_PROTECTED_BLOCKERS)
    ):
        raise MaterializationError("retained protected blocker binding differs")
    for alias, construction_status, completion_mode in (
        ("LGCVF-121", "blocked_external_authority", "external-authority"),
        ("LGCVF-123", "blocked_manual", "manual"),
    ):
        task = blocker_tasks.get(alias)
        body = task.get("body") if isinstance(task, Mapping) else None
        if (
            not isinstance(task, Mapping)
            or task.get("task_cid") != task_cids.get(alias)
            or task.get("status") != "blocked"
            or not isinstance(body, Mapping)
            or body.get("construction_status") != construction_status
            or body.get("completion") != completion_mode
            or body.get("is_schedulable") is not False
            or body.get("review_only") is not True
            or not str(body.get("blocked_reason") or "").startswith(
                construction_status + ";"
            )
        ):
            raise MaterializationError(f"{alias}: retained protected blocker differs")
    logical = binding.get("logical_completions")
    states = binding.get("task_states")
    dynamic = binding.get("completion_receipts")
    if (
        not isinstance(logical, list)
        or not isinstance(states, list)
        or not isinstance(dynamic, list)
        or len(logical) != 12
        or len(states) != 12
        or len(dynamic) != 5
        or {str(item.get("task_cid") or "") for item in logical} != expected_cids
        or {str(item.get("task_cid") or "") for item in states} != expected_cids
        or any(item.get("status") != "succeeded" for item in logical)
        or any(item.get("status") != "completed" for item in states)
    ):
        raise MaterializationError("retained completion population differs")
    construction_cids = {task_cids[item] for item in FRESH_RECOVERY_CONSTRUCTION_COMPLETIONS}
    dynamic_cids = {task_cids[item] for item in FRESH_RECOVERY_MERGE_COMPLETIONS[:-1]}
    for item in logical:
        task_cid = str(item.get("task_cid") or "")
        body = item.get("body")
        if not isinstance(body, Mapping):
            raise MaterializationError("retained logical completion body is malformed")
        if task_cid in construction_cids:
            if body.get("authority") != "database_population":
                raise MaterializationError("construction completion authority differs")
        elif (
            task_cid not in dynamic_cids
            or body.get("schema")
            != "ipfs_accelerate_py/agent-supervisor/task-completion-preparation@1"
        ):
            raise MaterializationError("dynamic retained completion authority differs")
    if {str(item.get("task_cid") or "") for item in dynamic} != dynamic_cids:
        raise MaterializationError("retained dynamic completion receipt population differs")
    for item in dynamic:
        validation = (
            item.get("body", {}).get("receipt", {}).get("validation", {})
            if isinstance(item.get("body"), Mapping)
            else {}
        )
        if validation.get("outcome") != "passed" or not str(
            validation.get("evidence_digest") or ""
        ).startswith("sha256:"):
            raise MaterializationError("retained dynamic validation evidence differs")
    return {
        "path": path.relative_to(source_root).as_posix(),
        "sha256": receipt_sha256,
        "receipt_cid": receipt["receipt_cid"],
        "successor_revision_cid": successor_revision,
        "delta_cid": delta_cid,
        "binding_cid": claimed,
        "protected_blocker_binding_cid": blocker_claimed,
        "construction_completion_count": 7,
        "dynamic_completion_receipt_count": 5,
        "logical_completion_count": 12,
    }


def _validate_wrong_default_quarantine(
    *,
    config: Mapping[str, Any],
    policy: Mapping[str, Any],
    source_root: Path,
    task_cids_by_alias: Mapping[str, str],
) -> dict[str, Any]:
    expected_contaminated_cids = dict(task_cids_by_alias)
    predecessor = _load_predecessor_plan(config, root=source_root)
    predecessor_cids = {task.task_id: task.content_id for task in predecessor.tasks}
    expected_contaminated_cids["LGCVF-120"] = predecessor_cids["LGCVF-120"]
    path, incident_bytes, observed_sha = _read_regular_evidence_bytes(
        source_root,
        policy.get("wrong_default_incident_manifest_path"),
        field="fresh_generation_recovery.wrong_default_incident_manifest_path",
    )
    if observed_sha != policy.get("wrong_default_incident_manifest_sha256"):
        raise MaterializationError("wrong-default incident manifest digest differs")
    incident = _decode_evidence_json(
        incident_bytes, noun="wrong-default incident manifest"
    )
    if set(incident) != {
        "schema",
        "manifest_cid",
        "preserved_under",
        "source_paths",
        "portal_entries",
        "portal_tree_cid",
        "todo",
    }:
        raise MaterializationError("wrong-default incident manifest fields differ")
    claimed = str(incident.pop("manifest_cid", ""))
    observed = content_identity(incident)
    incident["manifest_cid"] = claimed
    if (
        claimed != policy.get("wrong_default_incident_manifest_cid")
        or claimed != observed
        or incident.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/wrong-default-reload-quarantine@1"
        or incident.get("preserved_under")
        != (
            str(policy.get("source_runtime_root") or "")
            + "/evidence/quarantine/wrong-default-reload"
        )
        or incident.get("source_paths")
        != [
            "data/portal_implementation",
            "docs/211_SERVICE_NAVIGATION_PORTAL_TODO.md",
        ]
    ):
        raise MaterializationError("wrong-default incident manifest CID differs")
    preserved = str(incident.get("preserved_under") or "")
    entries = incident.get("portal_entries")
    if not isinstance(entries, list):
        raise MaterializationError("wrong-default portal quarantine is incomplete")
    observed_entries, portal_file_bytes = _snapshot_manifest_tree(
        source_root,
        Path(preserved) / "portal_implementation",
        entries,
        noun="wrong-default portal quarantine",
    )
    if (
        observed_entries != entries
        or content_identity(observed_entries) != incident.get("portal_tree_cid")
    ):
        raise MaterializationError("wrong-default portal entry closure differs")
    todo = incident.get("todo")
    _, todo_bytes, todo_sha256 = _read_regular_evidence_bytes(
        source_root,
        (Path(preserved) / "211_SERVICE_NAVIGATION_PORTAL_TODO.md").as_posix(),
        field="wrong-default quarantined TODO",
        expected_mode=str(todo.get("mode") or "") if isinstance(todo, Mapping) else "",
    )
    if (
        not isinstance(todo, Mapping)
        or set(todo) != {"kind", "mode", "path", "sha256", "size"}
        or todo.get("kind") != "file"
        or todo.get("path") != "docs/211_SERVICE_NAVIGATION_PORTAL_TODO.md"
        or len(todo_bytes) != int(todo.get("size") or 0)
        or todo_sha256 != todo.get("sha256")
    ):
        raise MaterializationError("wrong-default TODO quarantine identity differs")
    contamination_path, contamination_bytes, contamination_sha = (
        _read_regular_evidence_bytes(
        source_root,
        policy.get("contaminated_coordination_projection_path"),
        field="fresh_generation_recovery.contaminated_coordination_projection_path",
        )
    )
    if contamination_sha != policy.get("contaminated_coordination_projection_sha256"):
        raise MaterializationError("contaminated coordination manifest digest differs")
    contamination = _decode_evidence_json(
        contamination_bytes, noun="contaminated coordination manifest"
    )
    if set(contamination) != {
        "schema",
        "manifest_cid",
        "source_generation",
        "source_coordination_path",
        "source_projection_schema",
        "source_projection_root",
        "source_projection_counts",
        "rejected_task_ids",
        "rejected_completion_records",
        "rejected_record_set_cid",
        "rejection_reason",
        "disposition",
        "recovery_import_authority",
        "source_database_open_required_for_recovery",
    }:
        raise MaterializationError("contaminated coordination manifest fields differ")
    contamination_cid = str(contamination.pop("manifest_cid", ""))
    contamination_observed = content_identity(contamination)
    contamination["manifest_cid"] = contamination_cid
    records = contamination.get("rejected_completion_records")
    rejected_ids = contamination.get("rejected_task_ids")
    record_set_cid = (
        content_identity(records) if isinstance(records, list) else ""
    )
    rejected_root = str(contamination.get("source_projection_root") or "")
    expected_counts = {
        "active_fenced_leases": 0,
        "active_maintenance_leases": 0,
        "active_resource_claims": 0,
        "active_task_attempts": 0,
        "active_task_claims": 0,
        "dependency_edges": 46,
        "fenced_leases": 209,
        "logical_completions": 26,
        "maintenance_leases": 0,
        "registered_tasks": 28,
        "resource_claims": 0,
        "task_attempts": 209,
        "task_claims": 209,
    }
    if (
        contamination.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/lgcvf-contaminated-generation-projection@1"
        or contamination_cid != contamination_observed
        or contamination_cid
        != policy.get("contaminated_coordination_projection_manifest_cid")
        or record_set_cid != contamination.get("rejected_record_set_cid")
        or record_set_cid
        != policy.get("contaminated_coordination_rejected_record_set_cid")
        or rejected_ids != list(FRESH_RECOVERY_REJECTED_SYNTHETIC)
        or rejected_root
        != policy.get("rejected_contaminated_coordination_projection_root")
        or contamination.get("source_generation") != "lgcvf-run-v16"
        or contamination.get("source_projection_schema")
        != "ipfs_accelerate_py/agent-supervisor/coordination-registry-projection@1"
        or contamination.get("source_coordination_path")
        != (
            str(policy.get("source_runtime_root") or "")
            + "/control.coordination.duckdb"
        )
        or contamination.get("source_projection_counts") != expected_counts
        or contamination.get("rejection_reason")
        != (
            "synthetic database-validation completions without independently "
            "reproduced source mutation, declared validation, merge, or current-tree evidence"
        )
        or contamination.get("recovery_import_authority") is not False
        or contamination.get("source_database_open_required_for_recovery") is not False
        or contamination.get("disposition") != "forensic_quarantine"
    ):
        raise MaterializationError("contaminated coordination manifest binding differs")
    if not isinstance(records, list) or len(records) != 13:
        raise MaterializationError("contaminated completion record population differs")
    attempts: set[str] = set()
    claims: set[str] = set()
    leases: set[str] = set()
    for expected_id, record in zip(
        FRESH_RECOVERY_REJECTED_SYNTHETIC, records, strict=True
    ):
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {
                "task_id",
                "task_cid",
                "attempt_id",
                "attempt_number",
                "claim_id",
                "lease_id",
                "fencing_token",
                "fence_epoch",
                "logical_status",
                "preparation_digest",
                "validation_argv",
                "validation_outcome",
                "validation_evidence_digest",
                "control_revision",
                "control_completion_receipt_cid",
                "control_completion_receipt_digest",
            }
            or record.get("task_id") != expected_id
            or record.get("task_cid") != expected_contaminated_cids.get(expected_id)
            or record.get("logical_status") != "succeeded"
            or record.get("validation_argv") != ["database-validation"]
            or record.get("validation_outcome") != "passed"
            or record.get("attempt_number") != 1
            or record.get("fence_epoch") != 1
            or record.get("fencing_token") != 1
            or not _is_sha256(record.get("preparation_digest"))
            or not _is_sha256(record.get("validation_evidence_digest"))
            or not _is_sha256(record.get("control_completion_receipt_digest"))
            or not str(record.get("control_completion_receipt_cid") or "").startswith("b")
            or not str(record.get("attempt_id") or "").startswith("attempt:")
            or not str(record.get("claim_id") or "").startswith("claim:")
            or not str(record.get("lease_id") or "").startswith("lease:")
        ):
            raise MaterializationError("contaminated completion record identity differs")
        attempts.add(str(record["attempt_id"]))
        claims.add(str(record["claim_id"]))
        leases.add(str(record["lease_id"]))
    if len(attempts) != 13 or len(claims) != 13 or len(leases) != 13:
        raise MaterializationError("contaminated completion identities are duplicated")

    daemon_entries = [
        item
        for item in entries
        if isinstance(item, Mapping)
        and item.get("kind") == "file"
        and str(item.get("path") or "").startswith(
            "state/portal_implementation_daemon_"
        )
        and str(item.get("path") or "").endswith(".log")
    ]
    if len(daemon_entries) != 1:
        raise MaterializationError("wrong-default daemon log identity is ambiguous")
    daemon_relative = str(daemon_entries[0]["path"])
    try:
        raw_log = portal_file_bytes[daemon_relative].decode("utf-8")
    except (KeyError, UnicodeDecodeError) as exc:
        raise MaterializationError("wrong-default daemon log is unreadable") from exc
    if len(raw_log.encode("utf-8")) > 64 * 1024:
        raise MaterializationError("wrong-default daemon log exceeds its bound")
    marker = "Portal implementation daemon pass complete: "
    lines = raw_log.splitlines()
    if len(lines) != 14:
        raise MaterializationError("wrong-default daemon log event count differs")
    observed_log_records: list[dict[str, str]] = []
    for expected_id, record, line in zip(
        FRESH_RECOVERY_REJECTED_SYNTHETIC, records, lines[:13], strict=True
    ):
        if marker not in line:
            raise MaterializationError("wrong-default daemon log marker differs")
        try:
            event = ast.literal_eval(line.split(marker, 1)[1])
        except (SyntaxError, ValueError) as exc:
            raise MaterializationError("wrong-default daemon log event is malformed") from exc
        implementation = (
            event.get("implementation_result")
            if isinstance(event, Mapping)
            else None
        )
        attempt = (
            implementation.get("attempt")
            if isinstance(implementation, Mapping)
            else None
        )
        provider = (
            implementation.get("provider_result")
            if isinstance(implementation, Mapping)
            else None
        )
        effect = (
            implementation.get("effect_result")
            if isinstance(implementation, Mapping)
            else None
        )
        expected_attempt = str(record.get("attempt_id") or "")
        expected_cid = str(record.get("task_cid") or "")
        if (
            not isinstance(attempt, Mapping)
            or not isinstance(provider, Mapping)
            or not isinstance(effect, Mapping)
            or event.get("active_task_id") != expected_id
            or implementation.get("status") != "succeeded"
            or attempt.get("status") != "succeeded"
            or attempt.get("task_alias") != expected_id
            or attempt.get("task_cid") != expected_cid
            or attempt.get("attempt_id") != expected_attempt
            or attempt.get("claim_id") != record.get("claim_id")
            or provider
            != {
                "status": "noop",
                "provider": "database-implementation-daemon",
                "attempt_id": expected_attempt,
                "task_cid": expected_cid,
            }
            or effect.get("status") != "noop"
            or effect.get("effect") != "database-implementation-daemon"
            or effect.get("attempt_id") != expected_attempt
            or effect.get("task_cid") != expected_cid
            or effect.get("provider_result") != provider
        ):
            raise MaterializationError("wrong-default daemon log/record binding differs")
        observed_log_records.append(
            {
                "task_id": expected_id,
                "task_cid": expected_cid,
                "attempt_id": expected_attempt,
            }
        )
    if marker not in lines[-1]:
        raise MaterializationError("wrong-default daemon idle event is absent")
    try:
        idle = ast.literal_eval(lines[-1].split(marker, 1)[1])
    except (SyntaxError, ValueError) as exc:
        raise MaterializationError("wrong-default daemon idle event is malformed") from exc
    if idle != {"active_task_id": "", "selection_idle_reason": "no_ready_tasks"}:
        raise MaterializationError("wrong-default daemon did not close at no-ready-tasks")
    return {
        "incident_manifest_path": path.relative_to(source_root).as_posix(),
        "incident_manifest_sha256": observed_sha,
        "incident_manifest_cid": claimed,
        "portal_tree_cid": incident.get("portal_tree_cid"),
        "daemon_log_sha256": str(daemon_entries[0]["sha256"]),
        "daemon_log_rejected_record_binding_cid": content_identity(
            observed_log_records
        ),
        "contaminated_coordination_manifest_path": contamination_path.relative_to(
            source_root
        ).as_posix(),
        "contaminated_coordination_manifest_sha256": contamination_sha,
        "contaminated_coordination_manifest_cid": contamination_cid,
        "rejected_record_set_cid": record_set_cid,
        "rejected_contaminated_coordination_projection_root": rejected_root,
        "rejected_synthetic_task_ids": list(FRESH_RECOVERY_REJECTED_SYNTHETIC),
        "source_database_opened": False,
        "disposition": "preserved_forensic_quarantine_not_imported",
    }


def _validate_merge_completion(
    *,
    policy_item: Mapping[str, Any],
    task: Mapping[str, Any],
    source_root: Path,
    current_head: str,
) -> dict[str, Any]:
    task_id = str(task.get("task_id") or "")
    task_cid = str(task.get("task_cid") or "")
    if task_id != policy_item.get("task_id") or task_cid != policy_item.get("task_cid"):
        raise MaterializationError(f"{task_id}: recovery policy task identity differs")
    completed_path, completed_bytes, completed_sha256 = _read_regular_evidence_bytes(
        source_root,
        policy_item.get("completed_record_path"),
        field=f"fresh_generation_recovery.{task_id}.completed_record_path",
    )
    train_path, train_bytes, train_sha256 = _read_regular_evidence_bytes(
        source_root,
        policy_item.get("train_receipt_path"),
        field=f"fresh_generation_recovery.{task_id}.train_receipt_path",
    )
    if completed_sha256 != policy_item.get("completed_record_sha256"):
        raise MaterializationError(f"{task_id}: completed record digest differs")
    if train_sha256 != policy_item.get("train_receipt_sha256"):
        raise MaterializationError(f"{task_id}: train receipt digest differs")
    completed = _decode_evidence_json(
        completed_bytes, noun=f"{task_id} completed record"
    )
    train = _decode_evidence_json(train_bytes, noun=f"{task_id} train receipt")
    if set(completed) != {
        "attempt",
        "branch_name",
        "canonical_task_id",
        "canonical_task_key",
        "claim_generation",
        "claim_token",
        "claimed_at",
        "commit_sha",
        "consumer_id",
        "dedupe_key",
        "enqueued_at",
        "failure_count",
        "failure_reason",
        "lane_id",
        "metadata",
        "priority",
        "request_id",
        "retry_not_before",
        "status",
        "task_id",
    }:
        raise MaterializationError(f"{task_id}: completed record fields differ")
    if set(train) != {
        "acceptance_pending",
        "accepted",
        "callback_owned_integration",
        "canonical_task_id",
        "commit_sha",
        "distributed_publication_admission",
        "finished_at",
        "integrated",
        "merge_commit",
        "merge_result",
        "merged",
        "request_id",
        "started_at",
        "status",
        "target_branch",
        "target_commit",
        "task_id",
    }:
        raise MaterializationError(f"{task_id}: train receipt fields differ")
    request_id = str(policy_item.get("request_id") or "")
    candidate = str(policy_item.get("candidate_commit") or "")
    merge_commit = str(policy_item.get("merge_commit") or "")
    if (
        completed.get("request_id") != request_id
        or completed.get("task_id") != task_id
        or completed.get("canonical_task_id") != task_cid
        or completed.get("commit_sha") != candidate
        or completed.get("status") != "completed"
    ):
        raise MaterializationError(f"{task_id}: completed record binding differs")
    canonical_key = str(completed.get("canonical_task_key") or "")
    if (
        train.get("request_id") != request_id
        or train.get("task_id") != task_id
        or train.get("canonical_task_id") != canonical_key
        or train.get("commit_sha") != candidate
        or train.get("merge_commit") != merge_commit
        or train.get("target_commit") != merge_commit
        or train.get("target_branch")
        != "agent/logic-governed-compositional-verification-fabric-v1"
        or train.get("status") != "merged"
        or train.get("accepted") is not True
        or train.get("integrated") is not True
        or train.get("merged") is not True
        or train.get("acceptance_pending") is not False
    ):
        raise MaterializationError(f"{task_id}: merge train receipt binding differs")
    merge_result = train.get("merge_result")
    integration = (
        merge_result.get("integration_commit_proof")
        if isinstance(merge_result, Mapping)
        else None
    )
    output_invariant = (
        merge_result.get("post_merge_declared_output_invariant")
        if isinstance(merge_result, Mapping)
        else None
    )
    expected_outputs = sorted(
        str(item.get("path") or "")
        for item in task.get("outputs") or ()
        if isinstance(item, Mapping)
    )
    owner = str(task.get("owning_repository") or "")
    expected_train_outputs = sorted(
        "ipfs_datasets_py/" + item if owner == "ipfs_datasets_py" else item
        for item in expected_outputs
    )
    output_checks = (
        output_invariant.get("checks")
        if isinstance(output_invariant, Mapping)
        else None
    )
    if (
        not isinstance(merge_result, Mapping)
        or merge_result.get("attempted") is not True
        or merge_result.get("merged") is not True
        or merge_result.get("returncode") != 0
        or merge_result.get("merge_commit") != merge_commit
        or merge_result.get("target_branch")
        != "agent/logic-governed-compositional-verification-fabric-v1"
        or merge_result.get("generated_submodule_reconciliation") not in ([], ())
        or not isinstance(integration, Mapping)
        or integration.get("passed") is not True
        or integration.get("implementation_commit") != candidate
        or integration.get("integration_commit") != merge_commit
        or integration.get("integration_ref") != merge_commit
        or integration.get("target_branch")
        != "agent/logic-governed-compositional-verification-fabric-v1"
        or integration.get("reasons") not in ([], ())
        or not isinstance(output_invariant, Mapping)
        or output_invariant.get("passed") is not True
        or output_invariant.get("repository_ref") != merge_commit
        or output_invariant.get("task_ids") != [task_id]
        or output_invariant.get("missing_outputs") not in ([], ())
        or output_invariant.get("unsafe_outputs") not in ([], ())
        or output_invariant.get("untracked_outputs") not in ([], ())
        or not isinstance(output_checks, list)
        or sorted(
            str(item.get("path") or "")
            for item in output_checks
            if isinstance(item, Mapping)
        )
        != expected_train_outputs
        or len(output_checks) != len(expected_train_outputs)
        or any(
            not isinstance(item, Mapping)
            or item.get("task_id") != task_id
            or item.get("exists") is not True
            or item.get("tracked") is not True
            or item.get("reason") != "declared_output_tracked"
            for item in output_checks
        )
    ):
        raise MaterializationError(f"{task_id}: merge integration evidence differs")
    _git(source_root, "cat-file", "-e", candidate + "^{commit}")
    _git(source_root, "cat-file", "-e", merge_commit + "^{commit}")
    merge_parents = _git(
        source_root, "show", "-s", "--format=%P", merge_commit
    ).split()
    if len(merge_parents) != 2 or merge_parents[1] != candidate:
        raise MaterializationError(f"{task_id}: merge parent/candidate binding differs")
    _git(source_root, "merge-base", "--is-ancestor", candidate, merge_commit)
    _git(source_root, "merge-base", "--is-ancestor", merge_commit, current_head)
    metadata = completed.get("metadata")
    baseline = str(metadata.get("baseline_ref") or "") if isinstance(metadata, Mapping) else ""
    if owner == "ipfs_datasets_py":
        candidate_before = _gitlink_at(source_root, baseline, "ipfs_datasets_py")
        candidate_after = _gitlink_at(source_root, candidate, "ipfs_datasets_py")
        merged_after = _gitlink_at(source_root, merge_commit, "ipfs_datasets_py")
        submodules = merge_result.get("submodule_merge_results")
        submodule_invariant = merge_result.get("post_merge_submodule_invariant")
        invariant_paths = (
            submodule_invariant.get("paths")
            if isinstance(submodule_invariant, Mapping)
            else None
        )
        handoff = (
            submodule_invariant.get("integrated_handoff_proof")
            if isinstance(submodule_invariant, Mapping)
            else None
        )
        gitlink = merge_result.get("merged_gitlink_recording")
        if (
            candidate_after != merged_after
            or not isinstance(submodules, list)
            or len(submodules) != 1
            or not isinstance(submodules[0], Mapping)
            or submodules[0].get("path") != "ipfs_datasets_py"
            or submodules[0].get("merged") is not True
            or submodules[0].get("ancestry_valid") is not True
            or submodules[0].get("isolated_target") is not True
            or submodules[0].get("returncode") != 0
            or submodules[0].get("compare_and_swap_returncode") != 0
            or submodules[0].get("cleanup_returncode") != 0
            or submodules[0].get("commit") != candidate_after
            or submodules[0].get("branch_commit") != candidate_after
            or submodules[0].get("target_base_commit") != candidate_before
            or not isinstance(submodule_invariant, Mapping)
            or submodule_invariant.get("passed") is not True
            or submodule_invariant.get("candidate_commit") != candidate
            or submodule_invariant.get("target_commit") != merge_commit
            or not isinstance(invariant_paths, list)
            or len(invariant_paths) != 1
            or invariant_paths[0].get("path") != "ipfs_datasets_py"
            or invariant_paths[0].get("passed") is not True
            or invariant_paths[0].get("candidate_gitlink") != candidate_after
            or invariant_paths[0].get("target_gitlink") != candidate_after
            or not isinstance(handoff, Mapping)
            or handoff.get("passed") is not True
            or handoff.get("candidate_commit") != candidate
            or handoff.get("target_commit") != merge_commit
            or not isinstance(gitlink, Mapping)
            or gitlink.get("attempted") is not True
            or gitlink.get("ok") is not True
            or gitlink.get("failures") not in ([], ())
            or gitlink.get("expected_commits")
            != {"ipfs_datasets_py": candidate_after}
            or any(
                item.get("repository") != "ipfs_datasets_py"
                or item.get("repository_ref") != candidate_after
                for item in output_checks
            )
        ):
            raise MaterializationError(f"{task_id}: submodule merge evidence differs")
    else:
        if (
            merge_result.get("submodule_merge_results") not in ([], ())
            or merge_result.get("post_merge_submodule_invariant") is not None
            or merge_result.get("merged_gitlink_recording")
            != {
                "attempted": False,
                "committed": False,
                "ok": True,
                "reason": "no_matching_merged_root_gitlinks",
            }
            or any(
                item.get("repository") != "."
                or item.get("repository_ref") != merge_commit
                for item in output_checks
            )
        ):
            raise MaterializationError(f"{task_id}: unexpected submodule merge authority")
    changed_paths = _changed_paths_for_candidate(
        source_root=source_root,
        task=task,
        completed=completed,
        candidate_commit=candidate,
    )
    candidate_tree = _git(source_root, "show", "-s", "--format=%T", candidate)
    historical = _validate_historical_validation(
        task=task,
        completed=completed,
        candidate_commit=candidate,
        candidate_tree=candidate_tree,
    )
    return {
        "task_id": task_id,
        "task_cid": task_cid,
        "request_id": request_id,
        "completed_record_path": completed_path.relative_to(source_root).as_posix(),
        "completed_record_sha256": completed_sha256,
        "train_receipt_path": train_path.relative_to(source_root).as_posix(),
        "train_receipt_sha256": train_sha256,
        "candidate_commit": candidate,
        "merge_commit": merge_commit,
        "changed_paths": changed_paths,
        **historical,
    }


def preview_fresh_generation_recovery(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    root: Path = ROOT,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Validate immutable run-v16 evidence and render a no-write run-v17 import."""

    _require_isolated_recovery_interpreter()
    policy = _fresh_recovery_policy(config)
    duckdb_runtime_cid = _require_bound_duckdb_runtime_policy(config)
    authority_root = (source_root or root).resolve(strict=True)
    paths = _fresh_recovery_paths(config, root=root)
    lexical_target, lexical_target_state = _fresh_recovery_target_state(
        config, root=root
    )
    if paths["target"] != lexical_target:
        raise MaterializationError("fresh recovery target path resolution differs")
    current_head = _git(authority_root, "rev-parse", "HEAD")
    current_tree = _git(authority_root, "rev-parse", "HEAD^{tree}")
    if population.get("source_head") != current_head or population.get(
        "repository_tree_id"
    ) != "git-tree:" + current_tree:
        raise MaterializationError("fresh recovery population is stale for the current tree")
    _require_canonical_fresh_recovery_population(
        config,
        population,
        source_root=authority_root,
        current_head=current_head,
        current_tree=current_tree,
    )
    by_alias = {
        str(item.get("task_id") or ""): item
        for item in population.get("tasks") or ()
        if isinstance(item, Mapping)
    }
    if set(by_alias) != set(
        FRESH_RECOVERY_CONSTRUCTION_COMPLETIONS
        + FRESH_RECOVERY_MERGE_COMPLETIONS
        + FRESH_RECOVERY_REJECTED_SYNTHETIC
        + FRESH_RECOVERY_PROTECTED_BLOCKERS
    ):
        raise MaterializationError("fresh recovery task closure differs")
    for task_id in FRESH_RECOVERY_CONSTRUCTION_COMPLETIONS:
        if by_alias[task_id].get("status") != "completed":
            raise MaterializationError(f"{task_id}: construction completion differs")
    for task_id in FRESH_RECOVERY_MERGE_COMPLETIONS + FRESH_RECOVERY_REJECTED_SYNTHETIC:
        if by_alias[task_id].get("status") != "todo":
            raise MaterializationError(f"{task_id}: canonical pre-recovery state differs")
    for task_id in FRESH_RECOVERY_PROTECTED_BLOCKERS:
        if by_alias[task_id].get("status") != "blocked":
            raise MaterializationError(f"{task_id}: protected blocker differs")
    retained = _validate_retained_completion_binding(
        config=config,
        policy=policy,
        population=population,
        source_root=authority_root,
    )
    quarantine = _validate_wrong_default_quarantine(
        config=config,
        policy=policy,
        source_root=authority_root,
        task_cids_by_alias={
            task_id: str(task.get("task_cid") or "")
            for task_id, task in by_alias.items()
        },
    )
    raw_completions = policy.get("merge_completions")
    if not isinstance(raw_completions, list):
        raise MaterializationError("fresh recovery merge completion inventory is absent")
    evidence = [
        _validate_merge_completion(
            policy_item=item,
            task=by_alias[str(item.get("task_id") or "")],
            source_root=authority_root,
            current_head=current_head,
        )
        for item in raw_completions
        if isinstance(item, Mapping)
    ]
    if len(evidence) != 6:
        raise MaterializationError("fresh recovery did not validate six merge completions")
    target_state = lexical_target_state
    preview = {
        "schema": FRESH_RECOVERY_PREVIEW_SCHEMA,
        "write_performed": False,
        "source_generation": policy["source_generation"],
        "target_generation": policy["target_generation"],
        "duckdb_runtime_cid": duckdb_runtime_cid,
        "verification_python_executable": policy[
            "verification_python_executable"
        ],
        "verification_python_executable_sha256": policy[
            "verification_python_executable_sha256"
        ],
        "source_runtime_root": policy["source_runtime_root"],
        "target_runtime_root": policy["target_runtime_root"],
        "target_state": target_state,
        "source_head": current_head,
        "source_tree": current_tree,
        "plan_root_cid": population["plan_root_cid"],
        "population_root": population["population_root"],
        "retained_completion_binding": retained,
        "wrong_default_quarantine": quarantine,
        "merge_completion_evidence": evidence,
        "completion_partition": {
            "construction_completed_task_ids": list(
                FRESH_RECOVERY_CONSTRUCTION_COMPLETIONS
            ),
            "recovered_completed_task_ids": list(FRESH_RECOVERY_MERGE_COMPLETIONS),
            "rejected_synthetic_task_ids": list(FRESH_RECOVERY_REJECTED_SYNTHETIC),
            "preserved_blocked_task_ids": list(FRESH_RECOVERY_PROTECTED_BLOCKERS),
            "completed_count": 13,
            "todo_count": 13,
            "blocked_count": 2,
        },
        "synthetic_source_disposition": "quarantined_not_imported",
        "model_provider_route": "none",
        "validation_execution_state": "not_run",
        "validations_rerun": False,
    }
    preview["preview_cid"] = content_identity(preview)
    return preview


def _verify_recovery_qualification(
    value: Mapping[str, Any], *, preview: Mapping[str, Any], source_root: Path
) -> dict[str, Any]:
    """Verify one protected-judge receipt and bind it to admitted merge evidence."""

    try:
        from scripts.qualify_logic_governed_compositional_verification_fabric import (
            QualificationError,
            verify_preregistered_recovery_qualification,
        )
    except ImportError as exc:
        raise MaterializationError(
            "independent recovery qualifier API is unavailable"
        ) from exc
    try:
        verified = verify_preregistered_recovery_qualification(
            value, root=source_root, require_passed=True
        )
    except (OSError, QualificationError, TypeError, ValueError) as exc:
        raise MaterializationError(
            "independent recovery qualification is absent or invalid"
        ) from exc
    qualification = _plain_json(verified)
    if not isinstance(qualification, dict):
        raise MaterializationError("independent recovery qualification is malformed")
    observations = qualification.get("suites")
    merge_evidence = preview.get("merge_completion_evidence")
    if not isinstance(observations, list) or not isinstance(merge_evidence, list):
        raise MaterializationError("independent recovery observations are absent")
    if len(observations) != 6 or len(merge_evidence) != 6:
        raise MaterializationError("independent recovery observation count differs")
    for evidence, observation in zip(merge_evidence, observations, strict=True):
        if not isinstance(evidence, Mapping) or not isinstance(observation, Mapping):
            raise MaterializationError("independent recovery observation is malformed")
        spec = evidence.get("validation_spec")
        isolation = observation.get("isolation")
        if (
            not isinstance(spec, Mapping)
            or observation.get("schema")
            != "lgcvf-independent-recovery-pytest-observation@3"
            or observation.get("task_id") != spec.get("task_id")
            or observation.get("task_cid") != spec.get("task_cid")
            or observation.get("validation_spec") != dict(spec)
            or observation.get("passed") is not True
            or observation.get("exit_code") != 0
            or observation.get("cache_reused") is not False
            or observation.get("candidate_authored") is not True
            or observation.get("self_authority") is not False
            or observation.get("completion_authoritative") is not False
            or observation.get("provider_imports_observed") not in ([], ())
            or observation.get("provider_import_attempts") not in ([], ())
            or observation.get("provider_process_attempts") not in ([], ())
            or not isinstance(isolation, Mapping)
            or isolation.get("network_permitted") is not False
            or isolation.get("checkout_write_permitted") is not False
            or not str(observation.get("observation_cid") or "").startswith("b")
        ):
            raise MaterializationError(
                f"{spec.get('task_id') if isinstance(spec, Mapping) else 'unknown'}: "
                "independent recovery observation differs"
            )
    for field in (
        "task_implementation_complete",
        "objective_complete",
        "release_qualified",
        "production_authorized",
        "production_authoritative",
        "completion_authoritative",
    ):
        if qualification.get(field) is not False:
            raise MaterializationError(f"recovery qualification raises {field}")
    if (
        qualification.get("schema")
        != "lgcvf-independent-recovery-qualification@3"
        or qualification.get("passed") is not True
        or qualification.get("source_unchanged") is not True
        or qualification.get("exact_suite_membership") is not True
        or qualification.get("provider_route") != "none"
        or qualification.get("network_permitted") is not False
        or qualification.get("cache_reused") is not False
        or qualification.get("candidate_authored_replay") is not True
        or qualification.get("self_authority") is not False
        or not str(qualification.get("receipt_cid") or "").startswith("b")
    ):
        raise MaterializationError("recovery qualification test authorship differs")
    source_binding = qualification.get("source_binding_before")
    toolchain = source_binding.get("toolchain") if isinstance(source_binding, Mapping) else None
    duckdb_runtime = (
        toolchain.get("duckdb_runtime") if isinstance(toolchain, Mapping) else None
    )
    if (
        not isinstance(source_binding, Mapping)
        or source_binding.get("schema") != "lgcvf-recovery-source-binding@2"
        or not isinstance(duckdb_runtime, Mapping)
        or duckdb_runtime.get("schema") != "lgcvf-bound-duckdb-runtime@1"
        or duckdb_runtime.get("runtime_cid") != preview.get("duckdb_runtime_cid")
        or toolchain.get("python_executable")
        != preview.get("verification_python_executable")
        or toolchain.get("python_executable_sha256")
        != preview.get("verification_python_executable_sha256")
        or source_binding.get("accelerator_head") != preview.get("source_head")
        or source_binding.get("accelerator_tree") != preview.get("source_tree")
        or source_binding.get("datasets_gitlink")
        != source_binding.get("datasets_head")
        or qualification.get("plan_cid") != preview.get("plan_root_cid")
    ):
        raise MaterializationError("recovery qualification source binding differs")
    omission = qualification.get("validation_projection_omission_commitment")
    omission_root = qualification.get("validation_projection_omission_root")
    projection_evidence = qualification.get(
        "validation_projection_evidence_commitment"
    )
    projection_evidence_root = qualification.get(
        "validation_projection_evidence_root"
    )
    if (
        not isinstance(omission, Mapping)
        or set(omission)
        != {
            "schema",
            "accelerator_head",
            "accelerator_tree",
            "datasets_gitlink",
            "datasets_tree",
            "omitted_source_symlinks",
            "commitment_cid",
        }
        or omission.get("schema")
        != "lgcvf-recovery-validation-projection-omission@1"
        or omission.get("accelerator_head") != preview.get("source_head")
        or omission.get("accelerator_tree") != preview.get("source_tree")
        or omission.get("datasets_gitlink")
        != source_binding.get("datasets_gitlink")
        or omission.get("datasets_tree") != source_binding.get("datasets_tree")
        or omission.get("commitment_cid")
        != content_identity(
            {key: item for key, item in omission.items() if key != "commitment_cid"}
        )
        or omission_root != omission.get("commitment_cid")
    ):
        raise MaterializationError("recovery projection omission binding differs")
    omissions = omission.get("omitted_source_symlinks")
    if not isinstance(omissions, list):
        raise MaterializationError("recovery projection omission set is absent")
    observed_omissions: list[dict[str, Any]] = []
    for item in omissions:
        if not isinstance(item, Mapping) or set(item) != {
            "scope",
            "path",
            "git_target",
            "disposition",
        }:
            raise MaterializationError("recovery projection omission fields differ")
        path = str(item.get("path") or "")
        logical = Path(path)
        expected_scope = (
            "datasets_gitlink"
            if path.startswith("ipfs_datasets_py/")
            else "accelerator"
        )
        if (
            not path
            or logical.is_absolute()
            or ".." in logical.parts
            or logical.as_posix() != path
            or item.get("scope") != expected_scope
            or not isinstance(item.get("git_target"), str)
            or not item.get("git_target")
            or item.get("disposition") != "omitted_source_symlink"
        ):
            raise MaterializationError("recovery projection omission entry differs")
        observed_omissions.append(dict(item))
    if observed_omissions != sorted(
        observed_omissions, key=lambda item: (item["scope"], item["path"])
    ) or len({item["path"] for item in observed_omissions}) != len(
        observed_omissions
    ):
        raise MaterializationError("recovery projection omission ordering differs")
    expected_suites: list[dict[str, Any]] = []
    for observation in observations:
        projection = observation.get("readonly_projection")
        if (
            not isinstance(projection, Mapping)
            or projection.get("schema")
            != "lgcvf-closed-recovery-test-projection@1"
            or projection.get("contains_live_source_links") is not False
            or projection.get("original_checkout_writable") is not False
        ):
            raise MaterializationError("recovery closed-copy projection differs")
        expected_suites.append(
            {
                "suite_id": observation.get("suite_id"),
                "task_id": observation.get("task_id"),
                "task_cid": observation.get("task_cid"),
                "projection_cid": projection.get("projection_cid"),
                "copied_source_manifest_root": projection.get(
                    "copied_source_manifest_root"
                ),
            }
        )
    if (
        not isinstance(projection_evidence, Mapping)
        or set(projection_evidence)
        != {
            "schema",
            "source_binding_cid",
            "omission_root",
            "ordered_suites",
            "commitment_cid",
        }
        or projection_evidence.get("schema")
        != "lgcvf-recovery-validation-projection-evidence@1"
        or projection_evidence.get("source_binding_cid")
        != source_binding.get("source_binding_cid")
        or projection_evidence.get("omission_root") != omission_root
        or projection_evidence.get("ordered_suites") != expected_suites
        or projection_evidence.get("commitment_cid")
        != content_identity(
            {
                key: item
                for key, item in projection_evidence.items()
                if key != "commitment_cid"
            }
        )
        or projection_evidence_root != projection_evidence.get("commitment_cid")
    ):
        raise MaterializationError("recovery projection evidence binding differs")
    return qualification


def _run_and_verify_recovery_qualification(
    *, preview: Mapping[str, Any], source_root: Path
) -> dict[str, Any]:
    """Run only the closed OS-isolated public qualifier; no result injection."""

    try:
        from scripts.qualify_logic_governed_compositional_verification_fabric import (
            QualificationError,
            run_preregistered_recovery_qualification,
        )
    except ImportError as exc:
        raise MaterializationError(
            "independent recovery qualifier API is unavailable"
        ) from exc
    try:
        observed = run_preregistered_recovery_qualification(root=source_root)
    except (OSError, QualificationError, TypeError, ValueError) as exc:
        raise MaterializationError(
            "independent recovery qualification could not execute fail closed"
        ) from exc
    if not isinstance(observed, Mapping):
        raise MaterializationError("independent recovery qualifier returned no receipt")
    return _verify_recovery_qualification(
        observed, preview=preview, source_root=source_root
    )


def _require_canonical_fresh_recovery_population(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    source_root: Path,
    current_head: str,
    current_tree: str,
) -> dict[str, Any]:
    """Rebuild the exact revision-2 population from source-owned projections."""

    canonical_config = load_config()
    if _plain_json(config) != canonical_config:
        raise MaterializationError(
            "fresh recovery configuration differs from the canonical profile"
        )
    source = _project_source_binding(
        config,
        root=source_root,
        require_clean=False,
    )
    if (
        source.get("accelerator_head") != current_head
        or source.get("accelerator_tree") != current_tree
    ):
        raise MaterializationError("fresh recovery source topology changed")
    _formal_path, formal_bytes, _formal_digest = _read_regular_evidence_bytes(
        source_root,
        config.get("formal_plan_path"),
        field="formal_plan_path",
    )
    _todo_path, todo_bytes, _todo_digest = _read_regular_evidence_bytes(
        source_root,
        config.get("taskboard_path"),
        field="taskboard_path",
    )
    _require_head_bound_recovery_bytes(
        source_root,
        config.get("formal_plan_path"),
        formal_bytes,
        field="formal_plan_path",
    )
    _require_head_bound_recovery_bytes(
        source_root,
        config.get("taskboard_path"),
        todo_bytes,
        field="taskboard_path",
    )
    try:
        formal_plan = FormalWorkPlan.from_dict(
            _decode_evidence_json(formal_bytes, noun="LGCVF formal plan")
        )
        todo_text = todo_bytes.decode("utf-8")
    except (TypeError, UnicodeDecodeError, ValueError) as exc:
        raise MaterializationError(
            "fresh recovery canonical population source is invalid"
        ) from exc
    expected = project_population(
        config,
        formal_plan=formal_plan,
        todo_text=todo_text,
        source={
            "accelerator_head": current_head,
            "accelerator_tree": current_tree,
            "source_forest_root": source["source_forest_root"],
        },
    )
    if _plain_json(population) != expected:
        raise MaterializationError(
            "fresh recovery population differs from canonical source projection"
        )
    return expected


def _require_clean_recovery_source(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    source_root: Path,
) -> tuple[str, str]:
    try:
        source = verify_source_binding(config, root=source_root)
    except MaterializationError as exc:
        raise MaterializationError(
            f"fresh recovery requires an exact clean source binding: {exc}"
        ) from exc
    head = str(source["accelerator_head"])
    tree = str(source["accelerator_tree"])
    if (
        population.get("source_head") != head
        or population.get("repository_tree_id") != "git-tree:" + tree
        or population.get("source_forest_root") != source.get("source_forest_root")
    ):
        raise MaterializationError("fresh recovery source changed after population binding")
    _require_canonical_fresh_recovery_population(
        config,
        population,
        source_root=source_root,
        current_head=head,
        current_tree=tree,
    )
    return head, tree


def _source_evidence_cid(preview: Mapping[str, Any]) -> str:
    return content_identity(
        {
            "source_generation": preview["source_generation"],
            "target_generation": preview["target_generation"],
            "duckdb_runtime_cid": preview["duckdb_runtime_cid"],
            "source_head": preview["source_head"],
            "source_tree": preview["source_tree"],
            "plan_root_cid": preview["plan_root_cid"],
            "population_root": preview["population_root"],
            "retained_completion_binding": preview["retained_completion_binding"],
            "wrong_default_quarantine": preview["wrong_default_quarantine"],
            "merge_completion_evidence": preview["merge_completion_evidence"],
            "completion_partition": preview["completion_partition"],
            "synthetic_source_disposition": preview[
                "synthetic_source_disposition"
            ],
        }
    )


def _build_fresh_recovery_manifest(
    preview: Mapping[str, Any],
    *,
    source_root: Path,
) -> dict[str, Any]:
    qualification = _run_and_verify_recovery_qualification(
        preview=preview, source_root=source_root
    )
    manifest = {
        "schema": FRESH_RECOVERY_MANIFEST_SCHEMA,
        "source_evidence_cid": _source_evidence_cid(preview),
        "duckdb_runtime_cid": preview["duckdb_runtime_cid"],
        "source_generation": preview["source_generation"],
        "target_generation": preview["target_generation"],
        "source_runtime_root": preview["source_runtime_root"],
        "target_runtime_root": preview["target_runtime_root"],
        "source_head": preview["source_head"],
        "source_tree": preview["source_tree"],
        "plan_root_cid": preview["plan_root_cid"],
        "population_root": preview["population_root"],
        "retained_completion_binding": preview["retained_completion_binding"],
        "wrong_default_quarantine": preview["wrong_default_quarantine"],
        "merge_completion_evidence": preview["merge_completion_evidence"],
        "validation_qualification": qualification,
        "validation_qualification_cid": qualification["receipt_cid"],
        "validation_projection_omission_commitment": qualification[
            "validation_projection_omission_commitment"
        ],
        "validation_projection_omission_root": qualification[
            "validation_projection_omission_root"
        ],
        "validation_projection_evidence_commitment": qualification[
            "validation_projection_evidence_commitment"
        ],
        "validation_projection_evidence_root": qualification[
            "validation_projection_evidence_root"
        ],
        "completion_partition": preview["completion_partition"],
        "synthetic_source_disposition": "quarantined_not_imported",
        "source_database_statuses_read": False,
        "source_database_completion_records_imported": False,
        "model_provider_route": "none",
        "network_isolation_enforced": True,
        "validation_cache_reused": False,
        "candidate_authored_validation": True,
        "validation_self_authority": False,
        "validation_completion_authoritative": False,
        "task_implementation_complete": False,
        "test_qualification_complete": False,
        "objective_complete": False,
        "release_qualified": False,
        "production_authorized": False,
        "authority": "independent_merge_and_current_source_reconstruction",
    }
    manifest["manifest_cid"] = content_identity(manifest)
    return manifest


def _recovered_evidence_body(
    manifest: Mapping[str, Any],
    evidence: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": FRESH_RECOVERED_EVIDENCE_SCHEMA,
        "manifest_cid": manifest["manifest_cid"],
        "source_generation": manifest["source_generation"],
        "source_evidence_cid": manifest["source_evidence_cid"],
        "merge_completion_evidence_cid": content_identity(evidence),
        "request_id": evidence["request_id"],
        "completed_record_sha256": evidence["completed_record_sha256"],
        "train_receipt_sha256": evidence["train_receipt_sha256"],
        "candidate_commit": evidence["candidate_commit"],
        "merge_commit": evidence["merge_commit"],
        "validation_qualification_cid": manifest["validation_qualification_cid"],
        "validation_observation_cid": observation["observation_cid"],
        "candidate_authored_validation": True,
        "validation_self_authority": False,
        "validation_completion_authoritative": False,
    }


def _recovered_completion_receipt(
    manifest: Mapping[str, Any],
    evidence: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    reconstruction_evidence_digest = content_identity(
        _recovered_evidence_body(manifest, evidence, observation)
    )
    receipt = {
        "schema": FRESH_RECOVERED_COMPLETION_SCHEMA,
        "authority": "independent_merge_and_current_source_reconstruction",
        "manifest_cid": manifest["manifest_cid"],
        "source_generation": manifest["source_generation"],
        "target_generation": manifest["target_generation"],
        "source_evidence_cid": manifest["source_evidence_cid"],
        "task_id": evidence["task_id"],
        "task_cid": evidence["task_cid"],
        "request_id": evidence["request_id"],
        "completed_record_sha256": evidence["completed_record_sha256"],
        "train_receipt_sha256": evidence["train_receipt_sha256"],
        "candidate_commit": evidence["candidate_commit"],
        "merge_commit": evidence["merge_commit"],
        "merge_completion_evidence_cid": content_identity(evidence),
        "validation_qualification_cid": manifest["validation_qualification_cid"],
        "validation_observation_cid": observation["observation_cid"],
        "reconstruction_evidence_digest": reconstruction_evidence_digest,
        "candidate_authored_validation": True,
        "validation_self_authority": False,
        "validation_completion_authoritative": False,
        "release_qualified": False,
        "production_authorized": False,
    }
    receipt["receipt_cid"] = content_identity(receipt)
    return receipt


def _recovered_logical_body(
    manifest: Mapping[str, Any],
    *,
    task_id: str,
    control_receipt_cid: str,
    completion_receipt_cid: str,
    observation_cid: str,
    reconstruction_evidence_digest: str,
) -> dict[str, Any]:
    return {
        "authority": "lgcvf_fresh_generation_recovery",
        "manifest_cid": manifest["manifest_cid"],
        "source_generation": manifest["source_generation"],
        "target_generation": manifest["target_generation"],
        "task_alias": task_id,
        "control_receipt_cid": control_receipt_cid,
        "recovery_receipt_cid": completion_receipt_cid,
        "validation_qualification_cid": manifest["validation_qualification_cid"],
        "validation_observation_cid": observation_cid,
        "reconstruction_evidence_digest": reconstruction_evidence_digest,
        "candidate_authored_validation": True,
        "validation_completion_authoritative": False,
    }


def _apply_recovered_completions(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> list[dict[str, Any]]:
    """Use existing typed stores to import six independently checked completions."""

    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        DatabaseCoordinator,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    paths = _paths(config, root=root)
    task_source = DatabaseTaskSource(paths["control"], install_schema=False)
    coordinator = DatabaseCoordinator(paths["coordination"])
    evidence_by_task = {
        str(item.get("task_id") or ""): item
        for item in manifest.get("merge_completion_evidence") or ()
        if isinstance(item, Mapping)
    }
    qualification = manifest.get("validation_qualification")
    if not isinstance(qualification, Mapping):
        raise MaterializationError("recovery qualification is absent")
    validation_by_task = {
        str(item.get("task_id") or ""): item
        for item in qualification.get("suites") or ()
        if isinstance(item, Mapping)
    }
    receipts: list[dict[str, Any]] = []
    try:
        coordinator.open()
        for task_id in FRESH_RECOVERY_MERGE_COMPLETIONS:
            task = task_source.get_task(task_id)
            evidence = evidence_by_task.get(task_id)
            validation = validation_by_task.get(task_id)
            if task is None or not isinstance(evidence, Mapping) or not isinstance(
                validation, Mapping
            ):
                raise MaterializationError(f"{task_id}: recovery input is incomplete")
            if task.status != "todo" or task.task_cid != evidence.get("task_cid"):
                raise MaterializationError(f"{task_id}: fresh target state differs")
            evidence_digest = str(validation.get("observation_cid") or "")
            evidence_body = _recovered_evidence_body(manifest, evidence, validation)
            reconstruction_evidence_digest = content_identity(evidence_body)
            task_source.record_evidence(
                task_cid=task.task_cid,
                evidence_kind="fresh_generation_recovery_reconstruction",
                digest=reconstruction_evidence_digest,
                body=evidence_body,
            )
            completion_receipt = _recovered_completion_receipt(
                manifest, evidence, validation
            )
            control = task_source.compare_and_set_status(
                task,
                task.revision,
                "completed",
                completion_receipt,
                evidence_digests=(reconstruction_evidence_digest,),
            )
            logical = coordinator.mark_task_complete(
                task.task_cid,
                status="succeeded",
                body=_recovered_logical_body(
                    manifest,
                    task_id=task_id,
                    control_receipt_cid=control.receipt_cid,
                    completion_receipt_cid=completion_receipt["receipt_cid"],
                    observation_cid=evidence_digest,
                    reconstruction_evidence_digest=reconstruction_evidence_digest,
                ),
            )
            if (
                logical.get("task_cid") != task.task_cid
                or logical.get("status") != "succeeded"
                or isinstance(logical.get("completed_at_ms"), bool)
                or not isinstance(logical.get("completed_at_ms"), int)
            ):
                raise MaterializationError(f"{task_id}: logical completion failed")
            receipts.append(
                {
                    "task_id": task_id,
                    "task_cid": task.task_cid,
                    "control_revision": control.revision,
                    "control_receipt_cid": control.receipt_cid,
                    "recovery_receipt_cid": completion_receipt["receipt_cid"],
                    "logical_completion_status": "succeeded",
                    "validation_qualification_cid": manifest[
                        "validation_qualification_cid"
                    ],
                    "validation_observation_cid": evidence_digest,
                    "reconstruction_evidence_digest": reconstruction_evidence_digest,
                }
            )
    finally:
        coordinator.close()
        task_source.close()
    return receipts


def _reconstruct_fresh_recovery_authority(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    logical_root: Path,
) -> dict[str, Any]:
    """Independently reconstruct the exact fresh control/coordination state.

    The reconstruction uses the existing typed task-source and coordinator in
    a private temporary directory.  It never opens the quarantined generation
    and gives strict replay a full-fidelity oracle for task specifications,
    relations, goal/plan bodies, registration bodies, and all 13 logical
    completions.
    """

    with tempfile.TemporaryDirectory(prefix="lgcvf-recovery-reconstruction-") as temporary:
        temporary_root = Path(temporary)
        bootstrap = _materialize_fresh_recovery_stage(
            config, population, root=temporary_root, logical_root=logical_root
        )
        imported = _apply_recovered_completions(
            config, population, manifest, root=temporary_root
        )
        if len(imported) != len(FRESH_RECOVERY_MERGE_COMPLETIONS):
            raise MaterializationError("recovery reconstruction import count differs")
        operational = _verify_read_only_core(
            config, population, root=temporary_root, expected_stage="live"
        )
        paths = _paths(config, root=temporary_root)
        return {
            "bootstrap": bootstrap,
            "plan_projection": operational["control"]["plan_projection"],
            "task_revision_histories": operational["control"][
                "task_revision_histories"
            ],
            "objective_revision_history": operational["control"][
                "objective_revision_history"
            ],
            "plan_revision_history": operational["control"][
                "plan_revision_history"
            ],
            "control_schema_verification": operational["control"][
                "schema_verification"
            ],
            "control_table_counts": operational["control"]["table_counts"],
            "control_relation_inventory": operational["control"][
                "relation_inventory"
            ],
            "control_catalog_projection": operational["control"][
                "catalog_projection"
            ],
            "control_residual_content_projection": operational["control"][
                "residual_content_projection"
            ],
            "semantic_event_stream_root": operational["control"][
                "semantic_event_stream_root"
            ],
            "semantic_events": operational["control"]["semantic_events"],
            "execution": operational["execution"],
            "coordination": _plain_json(
                read_coordination_registry_projection(paths["coordination"])
            ),
            "coordination_history": _plain_json(
                read_coordination_history_projection(paths["coordination"])
            ),
            "coordination_catalog_projection": operational["coordination"][
                "catalog_projection"
            ],
        }


def _load_fresh_recovery_artifact(
    path: Path,
    *,
    schema: str,
    identity_field: str,
    expected_fields: set[str] | frozenset[str],
    noun: str,
) -> dict[str, Any]:
    value = _load_recovery_json(path, noun=noun)
    _require_exact_fields(value, expected_fields, noun=noun)
    if value.get("schema") != schema:
        raise MaterializationError(f"{noun} schema differs")
    claimed = str(value.pop(identity_field, ""))
    observed = content_identity(value)
    value[identity_field] = claimed
    if not claimed or claimed != observed:
        raise MaterializationError(f"{noun} content identity differs")
    return value


def _validate_fresh_recovery_bootstrap(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    root: Path,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct the canonical revision-2 bootstrap receipt binding."""

    paths = _fresh_recovery_paths(config, root=root)
    bootstrap_path = paths["receipt"]
    expected_sha256 = str(receipt.get("bootstrap_receipt_sha256") or "")
    if not _is_sha256(expected_sha256) or _sha256_file(bootstrap_path) != expected_sha256:
        raise MaterializationError("fresh recovery bootstrap receipt bytes differ")
    bootstrap = _load_receipt(bootstrap_path)
    _require_exact_fields(
        bootstrap,
        {
            "schema",
            "authority_mode",
            "task_source_kind",
            "maximum_writer_processes",
            "quack_qualified",
            "schema_revision",
            "schema_profile",
            "semantic_truth_authority",
            "operational_coordination_authority",
            "population_root",
            "plan_root_cid",
            "repository_tree_id",
            "source_head",
            "database_paths",
            "schema_install",
            "materialization",
            "verification",
            "receipt_cid",
        },
        noun="fresh recovery bootstrap receipt",
    )
    if (
        bootstrap.get("receipt_cid") != receipt.get("bootstrap_receipt_cid")
        or bootstrap.get("schema") != SCHEMA
        or bootstrap.get("authority_mode") != "embedded"
        or bootstrap.get("task_source_kind") != "duckdb"
        or bootstrap.get("maximum_writer_processes") != 1
        or bootstrap.get("quack_qualified") is not False
        or bootstrap.get("schema_revision") != EXPECTED_SCHEMA_REVISION
        or bootstrap.get("schema_profile") != EXPECTED_SCHEMA_PROFILE
        or bootstrap.get("semantic_truth_authority") != "ipfs_datasets_py"
        or bootstrap.get("operational_coordination_authority")
        != "ipfs_accelerate_py"
        or bootstrap.get("population_root") != population.get("population_root")
        or bootstrap.get("plan_root_cid") != population.get("plan_root_cid")
        or bootstrap.get("repository_tree_id")
        != population.get("repository_tree_id")
        or bootstrap.get("source_head") != population.get("source_head")
    ):
        raise MaterializationError("fresh recovery bootstrap authority differs")
    expected_database_paths = {
        key: path.relative_to(root).as_posix()
        for key, path in sorted(_paths(config, root=root).items())
        if key != "receipt"
    }
    if bootstrap.get("database_paths") != expected_database_paths:
        raise MaterializationError("fresh recovery bootstrap database paths differ")
    verification = bootstrap.get("verification")
    materialization = bootstrap.get("materialization")
    if (
        not isinstance(verification, Mapping)
        or verification.get("valid") is not True
        or verification.get("verification_mode") != "read_only"
        or verification.get("expected_stage") != "initial"
        or verification.get("population_root") != population.get("population_root")
        or verification.get("plan_root_cid") != population.get("plan_root_cid")
        or verification.get("repository_tree_id")
        != population.get("repository_tree_id")
        or verification.get("stores_unchanged") is not True
        or not isinstance(materialization, Mapping)
        or materialization.get("bootstrap_completed_task_cids")
        != [
            str(item.get("task_cid") or "")
            for item in population.get("tasks") or ()
            if isinstance(item, Mapping)
            and str(item.get("task_id") or "")
            in FRESH_RECOVERY_CONSTRUCTION_COMPLETIONS
        ]
    ):
        raise MaterializationError("fresh recovery bootstrap population differs")
    return bootstrap


def _normalized_fresh_bootstrap(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate nested bootstrap identities and remove only clock-derived data."""

    normalized = _plain_json(value)
    normalized.pop("receipt_cid", None)
    schema_install = normalized.get("schema_install")
    if not isinstance(schema_install, dict) or set(schema_install) != {
        "schema",
        "from_version",
        "to_version",
        "receipts",
        "schema_fingerprint",
        "catalog_fingerprint",
        "changed",
    }:
        raise MaterializationError("fresh recovery schema install receipt differs")
    receipts = schema_install.get("receipts")
    if not isinstance(receipts, list) or len(receipts) != 1:
        raise MaterializationError("fresh recovery migration receipt differs")
    stable_receipts: list[dict[str, Any]] = []
    for receipt in receipts:
        if not isinstance(receipt, dict) or set(receipt) != {
            "schema",
            "receipt_cid",
            "version",
            "migration_id",
            "checksum",
            "application_version",
            "tool_version",
            "started_at",
            "finished_at",
            "outcome",
            "schema_fingerprint",
            "error_text",
        }:
            raise MaterializationError("fresh recovery migration receipt fields differ")
        claimed = str(receipt.get("receipt_cid") or "")
        identity_body = {key: item for key, item in receipt.items() if key != "receipt_cid"}
        if claimed != content_identity(identity_body):
            raise MaterializationError("fresh recovery migration receipt CID differs")
        stable_receipts.append(
            {
                key: item
                for key, item in receipt.items()
                if key not in {"receipt_cid", "started_at", "finished_at"}
            }
        )
    schema_install["receipts"] = stable_receipts
    verification = normalized.get("verification")
    if not isinstance(verification, dict):
        raise MaterializationError("fresh recovery bootstrap verification is absent")
    claimed_root = str(verification.pop("verification_root", ""))
    if not claimed_root or claimed_root != content_identity(verification):
        raise MaterializationError("fresh recovery bootstrap verification root differs")
    control = verification.get("control")
    if not isinstance(control, dict):
        raise MaterializationError("fresh recovery bootstrap control projection is absent")
    control.pop("event_stream_root", None)
    return normalized


@_with_bound_duckdb_runtime
def verify_fresh_generation_recovery(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    root: Path = ROOT,
    source_root: Path | None = None,
    runtime_authority_root: Path | None = None,
) -> dict[str, Any]:
    """Read-only verify the exact published run-v17 recovery population."""

    _require_isolated_recovery_interpreter()
    authority_root = (source_root or root).resolve(strict=True)
    logical_root = (runtime_authority_root or root).resolve(strict=True)
    _require_clean_recovery_source(
        config, population, source_root=authority_root
    )
    lexical_target, lexical_state = _fresh_recovery_target_state(config, root=root)
    paths = _fresh_recovery_paths(config, root=root)
    if paths["target"] != lexical_target or lexical_state != "present":
        raise MaterializationError("run-v17 target is absent or not a real directory")
    _validate_fresh_recovery_layout(config, root=root)
    before_authority = _directory_fingerprint(
        paths["target"], require_private=True
    )
    receipt = _load_fresh_recovery_artifact(
        paths["recovery_receipt"],
        schema=FRESH_RECOVERY_RECEIPT_SCHEMA,
        identity_field="receipt_cid",
        expected_fields=FRESH_RECOVERY_RECEIPT_FIELDS,
        noun="fresh recovery receipt",
    )
    manifest_cid = str(receipt.get("manifest_cid") or "")
    if not manifest_cid or "/" in manifest_cid or ".." in manifest_cid:
        raise MaterializationError("fresh recovery manifest CID is unsafe")
    manifest_path = paths["recovery"] / f"{manifest_cid}.manifest.json"
    _validate_fresh_recovery_layout(
        config, root=root, manifest_path=manifest_path
    )
    manifest = _load_fresh_recovery_artifact(
        manifest_path,
        schema=FRESH_RECOVERY_MANIFEST_SCHEMA,
        identity_field="manifest_cid",
        expected_fields=FRESH_RECOVERY_MANIFEST_FIELDS,
        noun="fresh recovery manifest",
    )
    if manifest.get("manifest_cid") != manifest_cid:
        raise MaterializationError("fresh recovery manifest/receipt binding differs")
    preview = preview_fresh_generation_recovery(
        config, population, root=root, source_root=authority_root
    )
    if manifest.get("source_evidence_cid") != _source_evidence_cid(preview):
        raise MaterializationError("fresh recovery source evidence became stale")
    manifest_projection = {
        "source_generation": preview["source_generation"],
        "target_generation": preview["target_generation"],
        "duckdb_runtime_cid": preview["duckdb_runtime_cid"],
        "source_runtime_root": preview["source_runtime_root"],
        "target_runtime_root": preview["target_runtime_root"],
        "source_head": preview["source_head"],
        "source_tree": preview["source_tree"],
        "plan_root_cid": preview["plan_root_cid"],
        "population_root": preview["population_root"],
        "retained_completion_binding": preview["retained_completion_binding"],
        "wrong_default_quarantine": preview["wrong_default_quarantine"],
        "merge_completion_evidence": preview["merge_completion_evidence"],
        "completion_partition": preview["completion_partition"],
        "synthetic_source_disposition": "quarantined_not_imported",
        "source_database_statuses_read": False,
        "source_database_completion_records_imported": False,
        "model_provider_route": "none",
        "network_isolation_enforced": True,
        "validation_cache_reused": False,
        "candidate_authored_validation": True,
        "validation_self_authority": False,
        "validation_completion_authoritative": False,
        "task_implementation_complete": False,
        "test_qualification_complete": False,
        "objective_complete": False,
        "release_qualified": False,
        "production_authorized": False,
        "authority": "independent_merge_and_current_source_reconstruction",
    }
    if any(manifest.get(key) != value for key, value in manifest_projection.items()):
        raise MaterializationError("fresh recovery manifest authority differs")
    raw_qualification = manifest.get("validation_qualification")
    if not isinstance(raw_qualification, Mapping):
        raise MaterializationError("fresh recovery qualification is absent")
    qualification = _verify_recovery_qualification(
        raw_qualification, preview=preview, source_root=authority_root
    )
    qualification_cid = str(qualification.get("receipt_cid") or "")
    projection_bindings = {
        field: qualification[field]
        for field in (
            "validation_projection_omission_commitment",
            "validation_projection_omission_root",
            "validation_projection_evidence_commitment",
            "validation_projection_evidence_root",
        )
    }
    if (
        manifest.get("validation_qualification_cid") != qualification_cid
        or receipt.get("validation_qualification_cid") != qualification_cid
        or any(
            manifest.get(field) != value or receipt.get(field) != value
            for field, value in projection_bindings.items()
        )
    ):
        raise MaterializationError("fresh recovery qualification binding differs")
    receipt_projection = {
        "source_generation": manifest["source_generation"],
        "target_generation": manifest["target_generation"],
        "source_head": manifest["source_head"],
        "source_tree": manifest["source_tree"],
        "plan_root_cid": manifest["plan_root_cid"],
        "population_root": manifest["population_root"],
        "manifest_cid": manifest_cid,
        "source_evidence_cid": manifest["source_evidence_cid"],
        "duckdb_runtime_cid": manifest["duckdb_runtime_cid"],
        "completed_task_ids": list(
            FRESH_RECOVERY_CONSTRUCTION_COMPLETIONS
            + FRESH_RECOVERY_MERGE_COMPLETIONS
        ),
        "todo_task_ids": list(FRESH_RECOVERY_REJECTED_SYNTHETIC),
        "blocked_task_ids": list(FRESH_RECOVERY_PROTECTED_BLOCKERS),
        "completed_count": 13,
        "todo_count": 13,
        "blocked_count": 2,
        "validation_qualification_cid": qualification_cid,
        **projection_bindings,
        "model_provider_route": "none",
        "network_isolation_enforced": True,
        "candidate_authored_validation": True,
        "validation_self_authority": False,
        "validation_completion_authoritative": False,
        "task_implementation_complete": False,
        "test_qualification_complete": False,
        "objective_complete": False,
        "release_qualified": False,
        "production_authorized": False,
        "source_database_statuses_read": False,
        "source_database_completion_records_imported": False,
        "synthetic_source_disposition": "quarantined_not_imported",
        "atomic_publish": True,
    }
    if any(receipt.get(key) != value for key, value in receipt_projection.items()):
        raise MaterializationError("fresh recovery receipt authority differs")
    bootstrap = _validate_fresh_recovery_bootstrap(
        config, population, root=root, receipt=receipt
    )
    reconstructed = _reconstruct_fresh_recovery_authority(
        config, population, manifest, logical_root=logical_root
    )
    expected_bootstrap = reconstructed.get("bootstrap")
    if not isinstance(expected_bootstrap, Mapping) or _normalized_fresh_bootstrap(
        bootstrap
    ) != _normalized_fresh_bootstrap(expected_bootstrap):
        raise MaterializationError("fresh recovery nested bootstrap authority differs")
    operational = _verify_read_only_core(
        config, population, root=root, expected_stage="live"
    )
    if operational.get("control", {}).get("plan_projection") != reconstructed.get(
        "plan_projection"
    ):
        raise MaterializationError("fresh recovery full intent projection differs")
    if operational.get("control", {}).get(
        "task_revision_histories"
    ) != reconstructed.get("task_revision_histories"):
        raise MaterializationError("fresh recovery task revision history differs")
    if operational.get("control", {}).get(
        "objective_revision_history"
    ) != reconstructed.get("objective_revision_history"):
        raise MaterializationError("fresh recovery objective revision history differs")
    if operational.get("control", {}).get(
        "plan_revision_history"
    ) != reconstructed.get("plan_revision_history"):
        raise MaterializationError("fresh recovery plan revision history differs")
    if operational.get("control", {}).get("table_counts") != reconstructed.get(
        "control_table_counts"
    ):
        raise MaterializationError("fresh recovery control relation counts differ")
    if operational.get("control", {}).get(
        "relation_inventory"
    ) != reconstructed.get("control_relation_inventory"):
        raise MaterializationError("fresh recovery control relation inventory differs")
    if operational.get("control", {}).get(
        "schema_verification"
    ) != reconstructed.get("control_schema_verification"):
        raise MaterializationError("fresh recovery control schema differs")
    if operational.get("control", {}).get(
        "catalog_projection"
    ) != reconstructed.get("control_catalog_projection"):
        raise MaterializationError("fresh recovery control catalog differs")
    if operational.get("control", {}).get(
        "residual_content_projection"
    ) != reconstructed.get("control_residual_content_projection"):
        raise MaterializationError("fresh recovery control residual content differs")
    if operational.get("coordination", {}).get(
        "catalog_projection"
    ) != reconstructed.get("coordination_catalog_projection"):
        raise MaterializationError("fresh recovery coordination catalog differs")
    if operational.get("control", {}).get(
        "semantic_event_stream_root"
    ) != reconstructed.get("semantic_event_stream_root"):
        observed_events = operational.get("control", {}).get("semantic_events")
        expected_events = reconstructed.get("semantic_events")
        differing = -1
        if isinstance(observed_events, list) and isinstance(expected_events, list):
            differing = next(
                (
                    index
                    for index, pair in enumerate(
                        zip(observed_events, expected_events, strict=False)
                    )
                    if pair[0] != pair[1]
                ),
                min(len(observed_events), len(expected_events)),
            )
        observed_difference = (
            observed_events[differing]
            if isinstance(observed_events, list)
            and 0 <= differing < len(observed_events)
            else content_identity(observed_events)
        )
        expected_difference = (
            expected_events[differing]
            if isinstance(expected_events, list)
            and 0 <= differing < len(expected_events)
            else content_identity(expected_events)
        )
        raise MaterializationError(
            "fresh recovery semantic event stream differs at "
            f"event {differing}: observed={observed_difference} "
            f"expected={expected_difference}"
        )
    if operational.get("execution") != reconstructed.get("execution"):
        raise MaterializationError("fresh recovery execution projection differs")
    control_tasks = operational.get("control", {}).get("tasks")
    if not isinstance(control_tasks, list):
        raise MaterializationError("fresh recovery read-only task projection is absent")
    tasks = {
        str(item.get("task_alias") or ""): item
        for item in control_tasks
        if isinstance(item, Mapping)
    }
    ready = set(operational.get("control", {}).get("ready_task_aliases") or ())
    expected_statuses = {
        **{item: "completed" for item in FRESH_RECOVERY_CONSTRUCTION_COMPLETIONS},
        **{item: "completed" for item in FRESH_RECOVERY_MERGE_COMPLETIONS},
        **{item: "todo" for item in FRESH_RECOVERY_REJECTED_SYNTHETIC},
        **{item: "blocked" for item in FRESH_RECOVERY_PROTECTED_BLOCKERS},
    }
    if set(tasks) != set(expected_statuses) or any(
        tasks[alias].get("status") != status
        for alias, status in expected_statuses.items()
    ):
        raise MaterializationError("fresh recovery control status partition differs")
    if ready != {"LGCVF-081"}:
        raise MaterializationError("fresh recovery ready frontier differs")
    coordination = read_coordination_registry_projection(paths["coordination"])
    coordination_history = read_coordination_history_projection(
        paths["coordination"]
    )
    if coordination != reconstructed.get("coordination"):
        raise MaterializationError("fresh recovery coordination projection differs")
    if coordination_history != reconstructed.get("coordination_history"):
        raise MaterializationError("fresh recovery coordination history differs")
    logical = {
        str(item.get("task_cid") or ""): item
        for item in coordination.get("logical_completions") or ()
        if isinstance(item, Mapping)
    }
    expected_completed_cids = {str(tasks[item].get("task_cid") or "") for item in (
        FRESH_RECOVERY_CONSTRUCTION_COMPLETIONS + FRESH_RECOVERY_MERGE_COMPLETIONS
    )}
    if set(logical) != expected_completed_cids:
        raise MaterializationError("fresh recovery logical completion set differs")
    merge_evidence = {
        str(item.get("task_id") or ""): item
        for item in manifest.get("merge_completion_evidence") or ()
        if isinstance(item, Mapping)
    }
    observations = {
        str(item.get("task_id") or ""): item
        for item in qualification.get("suites") or ()
        if isinstance(item, Mapping)
    }
    if set(merge_evidence) != set(FRESH_RECOVERY_MERGE_COMPLETIONS) or set(
        observations
    ) != set(FRESH_RECOVERY_MERGE_COMPLETIONS):
        raise MaterializationError("fresh recovery imported evidence set differs")
    population_tasks = {
        str(item.get("task_id") or ""): item
        for item in population.get("tasks") or ()
        if isinstance(item, Mapping)
    }
    control_evidence = operational.get("control", {}).get("evidence")
    control_completions = operational.get("control", {}).get(
        "completion_receipts"
    )
    if not isinstance(control_evidence, list) or not isinstance(
        control_completions, list
    ):
        raise MaterializationError("fresh recovery control evidence is absent")
    expected_evidence_rows: list[dict[str, Any]] = []
    expected_completion_rows: list[dict[str, Any]] = []
    expected_imports: list[dict[str, Any]] = []
    for alias in FRESH_RECOVERY_MERGE_COMPLETIONS:
        evidence = merge_evidence[alias]
        observation = observations[alias]
        task = tasks[alias]
        source_task = population_tasks[alias]
        task_cid = str(task.get("task_cid") or "")
        expected_completion = _recovered_completion_receipt(
            manifest, evidence, observation
        )
        body = task.get("body")
        if (
            task.get("revision") != 2
            or not isinstance(body, Mapping)
            or body.get("completion_receipt") != expected_completion
        ):
            raise MaterializationError(f"{alias}: recovered completion receipt differs")
        observation_cid = str(observation.get("observation_cid") or "")
        evidence_body = _recovered_evidence_body(manifest, evidence, observation)
        reconstruction_evidence_digest = content_identity(evidence_body)
        evidence_row_body = {
            "task_cid": task_cid,
            "evidence_kind": "fresh_generation_recovery_reconstruction",
            "digest": reconstruction_evidence_digest,
            "body": evidence_body,
        }
        expected_evidence_rows.append(
            {
                "evidence_id": content_identity(evidence_row_body),
                "parent_evidence_id": "",
                **evidence_row_body,
            }
        )
        control_evidence_digest = content_identity(
            {
                "task_cid": task_cid,
                "revision": 2,
                "receipt": expected_completion,
                "evidence_digests": [reconstruction_evidence_digest],
            }
        )
        control_receipt_cid = content_identity(
            {
                "namespace": "completion-receipt",
                "task_cid": task_cid,
                "revision": 2,
                "evidence_digest": control_evidence_digest,
            }
        )
        expected_completion_rows.append(
            {
                "receipt_cid": control_receipt_cid,
                "task_cid": task_cid,
                "goal_cid": str(source_task.get("goal_cid") or ""),
                "attempt_id": "",
                "claim_cid": "",
                "fencing_token": 0,
                "validation_run_id": "",
                "evidence_digest": control_evidence_digest,
                "body": {
                    "schema": "ipfs_accelerate_py/agent-supervisor/intent-completion-evidence@1",
                    "receipt": expected_completion,
                    "evidence_digests": [reconstruction_evidence_digest],
                    "revision": 2,
                },
            }
        )
        expected_logical_body = _recovered_logical_body(
            manifest,
            task_id=alias,
            control_receipt_cid=control_receipt_cid,
            completion_receipt_cid=str(expected_completion["receipt_cid"]),
            observation_cid=observation_cid,
            reconstruction_evidence_digest=reconstruction_evidence_digest,
        )
        if logical[task_cid] != {
            "task_cid": task_cid,
            "status": "succeeded",
            "body": expected_logical_body,
        }:
            raise MaterializationError(f"{alias}: logical recovery authority differs")
        expected_imports.append(
            {
                "task_id": alias,
                "task_cid": task_cid,
                "control_revision": 2,
                "control_receipt_cid": control_receipt_cid,
                "recovery_receipt_cid": expected_completion["receipt_cid"],
                "logical_completion_status": "succeeded",
                "validation_qualification_cid": qualification_cid,
                "validation_observation_cid": observation_cid,
                "reconstruction_evidence_digest": reconstruction_evidence_digest,
            }
        )
    if control_evidence != sorted(
        expected_evidence_rows, key=lambda item: (item["task_cid"], item["evidence_id"])
    ):
        raise MaterializationError("fresh recovery control evidence population differs")
    if control_completions != sorted(
        expected_completion_rows,
        key=lambda item: (item["task_cid"], item["receipt_cid"]),
    ):
        raise MaterializationError("fresh recovery control completion population differs")
    imported = receipt.get("imported_completions")
    if not isinstance(imported, list) or any(
        not isinstance(item, Mapping)
        or set(item) != FRESH_RECOVERY_IMPORTED_COMPLETION_FIELDS
        for item in imported
    ):
        raise MaterializationError("fresh recovery imported completion schema differs")
    if imported != expected_imports:
        raise MaterializationError("fresh recovery imported completion binding differs")
    expected_coordination_counts = {
        "registered_tasks": 28,
        "dependency_edges": sum(
            len(item.get("dependencies") or ())
            for item in population.get("tasks") or ()
            if isinstance(item, Mapping)
        ),
        "logical_completions": 13,
        "task_claims": 0,
        "active_task_claims": 0,
        "resource_claims": 0,
        "active_resource_claims": 0,
        "task_attempts": 0,
        "active_task_attempts": 0,
        "fenced_leases": 0,
        "active_fenced_leases": 0,
        "maintenance_leases": 0,
        "active_maintenance_leases": 0,
    }
    if coordination.get("counts") != expected_coordination_counts or any(
        coordination.get(field) != []
        for field in (
            "task_claims",
            "task_attempts",
            "fenced_leases",
            "resource_claims",
            "maintenance_leases",
        )
    ):
        raise MaterializationError("fresh recovery coordination history is not empty")
    if coordination_history.get("counts") != {
        "token_history": 0,
        "lease_events": 0,
    } or any(
        coordination_history.get(field) != []
        for field in ("token_history", "lease_events")
    ):
        raise MaterializationError("fresh recovery fencing history is not empty")
    if any(operational["execution"]["row_counts"].values()):
        raise MaterializationError("fresh recovery published execution attempts")
    execution_metadata = operational["execution"].get("metadata")
    control_schema_verification = operational["control"].get(
        "schema_verification"
    )
    logical_paths = _paths(config, root=logical_root)
    expected_owner_material = "\n".join(
        str(logical_paths[key].absolute())
        for key in ("control", "coordination", "execution")
    ).encode("utf-8")
    expected_logical_owner = (
        "embedded-store:"
        + hashlib.sha256(expected_owner_material).hexdigest()[:32]
    )
    if (
        not isinstance(execution_metadata, Mapping)
        or execution_metadata.get("process_instance_id")
        != "fresh-recovery-bootstrap"
        or execution_metadata.get("logical_owner_session_id")
        != expected_logical_owner
        or not isinstance(control_schema_verification, Mapping)
        or control_schema_verification.get("valid") is not True
        or execution_metadata.get("control_schema_profile_id")
        != control_schema_verification.get("profile_id")
        or execution_metadata.get("control_schema_fingerprint")
        != control_schema_verification.get("schema_fingerprint")
    ):
        raise MaterializationError("fresh recovery execution authority differs")
    if receipt.get("operational_verification_root") != operational.get(
        "verification_root"
    ):
        raise MaterializationError("fresh recovery operational root differs")
    _validate_fresh_recovery_layout(
        config, root=root, manifest_path=manifest_path
    )
    after_authority = _directory_fingerprint(
        paths["target"], require_private=True
    )
    _validate_fresh_recovery_layout(
        config, root=root, manifest_path=manifest_path
    )
    closed_authority = _directory_fingerprint(
        paths["target"], require_private=True
    )
    if before_authority != after_authority or after_authority != closed_authority:
        raise MaterializationError("strict recovery verification changed authority files")
    report = {
        "schema": FRESH_RECOVERY_VERIFICATION_SCHEMA,
        "valid": True,
        "verification_mode": "read_only",
        "source_generation": manifest["source_generation"],
        "target_generation": manifest["target_generation"],
        "manifest_cid": manifest_cid,
        "receipt_cid": receipt["receipt_cid"],
        "source_evidence_cid": manifest["source_evidence_cid"],
        "duckdb_runtime_cid": manifest["duckdb_runtime_cid"],
        "completed_task_ids": list(
            FRESH_RECOVERY_CONSTRUCTION_COMPLETIONS
            + FRESH_RECOVERY_MERGE_COMPLETIONS
        ),
        "todo_task_ids": list(FRESH_RECOVERY_REJECTED_SYNTHETIC),
        "blocked_task_ids": list(FRESH_RECOVERY_PROTECTED_BLOCKERS),
        "completed_count": 13,
        "todo_count": 13,
        "blocked_count": 2,
        "ready_task_ids": ["LGCVF-081"],
        "validation_qualification_cid": qualification_cid,
        **projection_bindings,
        "model_provider_route": "none",
        "network_isolation_enforced": True,
        "candidate_authored_validation": True,
        "validation_completion_authoritative": False,
        "task_implementation_complete": False,
        "test_qualification_complete": False,
        "objective_complete": False,
        "release_qualified": False,
        "production_authorized": False,
        "source_database_statuses_read": False,
        "synthetic_source_disposition": "quarantined_not_imported",
        "operational_verification_root": operational["verification_root"],
        "stores_unchanged": True,
    }
    report["verification_root"] = content_identity(report)
    return report


def _materialize_fresh_recovery_stage(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    root: Path,
    logical_root: Path,
) -> dict[str, Any]:
    """Build the canonical stage through provider-cold typed store APIs."""

    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        DatabaseCoordinator,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        install_datasets_authoritative_operational_schema,
        verify_datasets_authoritative_operational_schema,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_execution_schema import (
        install_database_execution_schema,
    )

    paths = _paths(config, root=root)
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise MaterializationError("fresh recovery stage is not empty")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        duckdb_version = importlib.metadata.version("duckdb")
    except importlib.metadata.PackageNotFoundError:
        duckdb_version = "unavailable"
    database_uuid = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            "ipfs_accelerate_py/lgcvf-fresh-recovery/"
            + str(_fresh_recovery_policy(config)["target_generation"])
            + "/"
            + str(population["plan_root_cid"])
            + "/"
            + str(population["population_root"]),
        )
    )
    schema_install = install_datasets_authoritative_operational_schema(
        paths["control"],
        application_version="lgcvf-v1",
        tool_version=duckdb_version,
        owner_id="lgcvf-materializer:operational-schema",
        database_uuid=database_uuid,
    )
    task_source = DatabaseTaskSource(
        paths["control"],
        owner_id="lgcvf-fresh-recovery:task-source",
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
        install_schema=False,
    )
    coordinator = DatabaseCoordinator(paths["coordination"])
    try:
        task_source_receipt = _plain_json(
            task_source.materialize(
                population,
                repository_tree_id=str(population["repository_tree_id"]),
                plan_root_cid=str(population["plan_root_cid"]),
            )
        )
        coordinator.open()
        registered: list[str] = []
        bootstrap_completed: list[str] = []
        for task in task_source.list_tasks(limit=1_000).tasks:
            coordinator.register_task(
                task_cid=task.task_cid,
                task_id=task.task_alias or task.task_cid,
                dependency_task_cids=tuple(str(dep) for dep in task.dependencies),
                body={
                    "task_alias": task.task_alias,
                    "status": task.status,
                    "priority": task.priority,
                },
            )
            registered.append(task.task_cid)
            status = str(task.status or "").strip().lower()
            if status in {"completed", "complete", "done"}:
                coordinator.mark_task_complete(
                    task.task_cid,
                    status="succeeded",
                    body={
                        "authority": "database_population",
                        "source_status": status,
                        "task_alias": task.task_alias,
                        "task_revision": int(task.revision),
                    },
                )
                bootstrap_completed.append(task.task_cid)
    finally:
        coordinator.close()
        task_source.close()
    control_verification = verify_datasets_authoritative_operational_schema(
        paths["control"]
    )
    logical_paths = _paths(config, root=logical_root)
    bootstrap_owner_material = "\n".join(
        str(logical_paths[key].absolute())
        for key in ("control", "coordination", "execution")
    ).encode("utf-8")
    bootstrap_owner = (
        "embedded-store:"
        + hashlib.sha256(bootstrap_owner_material).hexdigest()[:32]
    )
    install_database_execution_schema(
        paths["execution"],
        metadata={
            "authority_mode": "embedded",
            "logical_owner_session_id": bootstrap_owner,
            "process_instance_id": "fresh-recovery-bootstrap",
            "state_schema_revision": EXPECTED_SCHEMA_REVISION,
            "control_schema_profile_id": str(
                control_verification.get("profile_id") or ""
            ),
            "control_schema_fingerprint": str(
                control_verification.get("schema_fingerprint") or ""
            ),
        },
    )
    database_receipt = {
        "task_source": task_source_receipt,
        "registered_task_cids": registered,
        "bootstrap_completed_task_cids": bootstrap_completed,
    }
    verification = _verify_read_only_core(
        config, population, root=root, expected_stage="initial"
    )
    receipt = {
        "schema": SCHEMA,
        "authority_mode": "embedded",
        "task_source_kind": "duckdb",
        "maximum_writer_processes": 1,
        "quack_qualified": False,
        "schema_revision": EXPECTED_SCHEMA_REVISION,
        "schema_profile": EXPECTED_SCHEMA_PROFILE,
        "semantic_truth_authority": "ipfs_datasets_py",
        "operational_coordination_authority": "ipfs_accelerate_py",
        "population_root": population["population_root"],
        "plan_root_cid": population["plan_root_cid"],
        "repository_tree_id": population["repository_tree_id"],
        "source_head": population["source_head"],
        "database_paths": {
            key: path.relative_to(root).as_posix()
            for key, path in sorted(paths.items())
            if key != "receipt"
        },
        "schema_install": (
            schema_install.to_dict()
            if callable(getattr(schema_install, "to_dict", None))
            else dict(schema_install)
        ),
        "materialization": database_receipt,
        "verification": verification,
    }
    receipt["receipt_cid"] = content_identity(receipt)
    paths["receipt"].write_bytes(_canonical_bytes(receipt) + b"\n")
    return receipt


@_with_bound_duckdb_runtime
def materialize_fresh_generation_recovery(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    root: Path = ROOT,
    source_root: Path | None = None,
    fault_injector: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Atomically publish a new run-v17 containing only 13 proved completions."""

    _require_isolated_recovery_interpreter()
    authority_root = (source_root or root).resolve(strict=True)
    source_binding = _require_clean_recovery_source(
        config, population, source_root=authority_root
    )
    target, target_state = _fresh_recovery_target_state(config, root=root)
    paths = _fresh_recovery_paths(config, root=root)
    if paths["target"] != target:
        raise MaterializationError("fresh recovery target path resolution differs")
    if target_state == "present":
        verification = verify_fresh_generation_recovery(
            config, population, root=root, source_root=authority_root
        )
        receipt = _load_fresh_recovery_artifact(
            paths["recovery_receipt"],
            schema=FRESH_RECOVERY_RECEIPT_SCHEMA,
            identity_field="receipt_cid",
            expected_fields=FRESH_RECOVERY_RECEIPT_FIELDS,
            noun="fresh recovery receipt",
        )
        if verification.get("receipt_cid") != receipt.get("receipt_cid"):
            raise MaterializationError("fresh recovery replay verification differs")
        return receipt
    preview = preview_fresh_generation_recovery(
        config, population, root=root, source_root=authority_root
    )
    if preview.get("target_state") != "absent":
        raise MaterializationError("run-v17 must be wholly absent before recovery")
    if fault_injector is not None:
        fault_injector("after_static_validation")
    manifest = _build_fresh_recovery_manifest(
        preview,
        source_root=authority_root,
    )
    _require_clean_recovery_source(
        config, population, source_root=authority_root
    )
    target, target_state = _fresh_recovery_target_state(config, root=root)
    if target_state != "absent":
        raise MaterializationError("run-v17 appeared after recovery validation")
    staging_container = target.with_name(
        f"{target.name}-fresh-recovery-staging"
    )
    lock_path = staging_container / "recovery.lock"
    _require_git_ignored_recovery_paths(
        authority_root,
        (lock_path, staging_container / "stage-probe"),
    )
    parent_descriptor = _open_or_create_directory_chain(root, target.parent)
    staging_flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    staging_created = False
    try:
        os.mkdir(
            staging_container.name,
            mode=0o700,
            dir_fd=parent_descriptor,
        )
        staging_created = True
        os.fsync(parent_descriptor)
    except FileExistsError:
        pass
    try:
        staging_descriptor = os.open(
            staging_container.name,
            staging_flags,
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        os.close(parent_descriptor)
        raise MaterializationError(
            "fresh recovery staging directory cannot be opened safely"
        ) from exc
    staging_status = os.fstat(staging_descriptor)
    if (
        not stat.S_ISDIR(staging_status.st_mode)
        or staging_status.st_uid != os.geteuid()
        or stat.S_IMODE(staging_status.st_mode) != 0o700
    ):
        os.close(staging_descriptor)
        os.close(parent_descriptor)
        raise MaterializationError("fresh recovery staging identity differs")
    if staging_created:
        os.fsync(staging_descriptor)
    lock_flags = os.O_CREAT | os.O_RDWR
    lock_flags |= getattr(os, "O_CLOEXEC", 0)
    lock_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        lock_descriptor = os.open(
            lock_path.name, lock_flags, 0o600, dir_fd=staging_descriptor
        )
    except OSError as exc:
        os.close(staging_descriptor)
        os.close(parent_descriptor)
        raise MaterializationError("fresh recovery lock cannot be opened safely") from exc
    lock_status = os.fstat(lock_descriptor)
    if (
        not stat.S_ISREG(lock_status.st_mode)
        or lock_status.st_uid != os.geteuid()
        or stat.S_IMODE(lock_status.st_mode) != 0o600
        or lock_status.st_nlink != 1
    ):
        os.close(lock_descriptor)
        os.close(staging_descriptor)
        os.close(parent_descriptor)
        raise MaterializationError("fresh recovery lock identity differs")
    stage_root: Path | None = None
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MaterializationError("another fresh recovery writer is active") from exc
        _reconcile_stale_fresh_recovery_stages(staging_descriptor)
        observed_target, observed_state = _fresh_recovery_target_state(
            config, root=root
        )
        if observed_target != target or observed_state != "absent":
            raise MaterializationError("run-v17 appeared after recovery validation")
        stage_root = Path(
            tempfile.mkdtemp(
                prefix="stage-",
                dir=f"/proc/self/fd/{staging_descriptor}",
            )
        ).resolve(strict=True)
        _require_git_ignored_recovery_paths(authority_root, (stage_root,))
        bootstrap = _materialize_fresh_recovery_stage(
            config,
            population,
            root=stage_root,
            logical_root=root,
        )
        if fault_injector is not None:
            fault_injector("after_stage_materialization")
        imported = _apply_recovered_completions(
            config, population, manifest, root=stage_root
        )
        if len(imported) != 6:
            raise MaterializationError("fresh recovery import count differs")
        if fault_injector is not None:
            fault_injector("after_completion_import")
        operational = _verify_read_only_core(
            config, population, root=stage_root, expected_stage="live"
        )
        staged_paths = _fresh_recovery_paths(config, root=stage_root)
        manifest_path = (
            staged_paths["recovery"] / f"{manifest['manifest_cid']}.manifest.json"
        )
        _atomic_write_json(manifest_path, manifest)
        receipt = {
            "schema": FRESH_RECOVERY_RECEIPT_SCHEMA,
            "source_generation": manifest["source_generation"],
            "target_generation": manifest["target_generation"],
            "source_head": manifest["source_head"],
            "source_tree": manifest["source_tree"],
            "plan_root_cid": manifest["plan_root_cid"],
            "population_root": manifest["population_root"],
            "manifest_cid": manifest["manifest_cid"],
            "source_evidence_cid": manifest["source_evidence_cid"],
            "duckdb_runtime_cid": manifest["duckdb_runtime_cid"],
            "bootstrap_receipt_cid": bootstrap["receipt_cid"],
            "bootstrap_receipt_sha256": _sha256_file(staged_paths["receipt"]),
            "imported_completions": imported,
            "completed_task_ids": list(
                FRESH_RECOVERY_CONSTRUCTION_COMPLETIONS
                + FRESH_RECOVERY_MERGE_COMPLETIONS
            ),
            "todo_task_ids": list(FRESH_RECOVERY_REJECTED_SYNTHETIC),
            "blocked_task_ids": list(FRESH_RECOVERY_PROTECTED_BLOCKERS),
            "completed_count": 13,
            "todo_count": 13,
            "blocked_count": 2,
            "validation_qualification_cid": manifest[
                "validation_qualification_cid"
            ],
            "validation_projection_omission_commitment": manifest[
                "validation_projection_omission_commitment"
            ],
            "validation_projection_omission_root": manifest[
                "validation_projection_omission_root"
            ],
            "validation_projection_evidence_commitment": manifest[
                "validation_projection_evidence_commitment"
            ],
            "validation_projection_evidence_root": manifest[
                "validation_projection_evidence_root"
            ],
            "model_provider_route": "none",
            "network_isolation_enforced": True,
            "candidate_authored_validation": True,
            "validation_self_authority": False,
            "validation_completion_authoritative": False,
            "task_implementation_complete": False,
            "test_qualification_complete": False,
            "objective_complete": False,
            "release_qualified": False,
            "production_authorized": False,
            "source_database_statuses_read": False,
            "source_database_completion_records_imported": False,
            "synthetic_source_disposition": "quarantined_not_imported",
            "operational_verification_root": operational["verification_root"],
            "atomic_publish": True,
        }
        receipt["receipt_cid"] = content_identity(receipt)
        _atomic_write_json(staged_paths["recovery_receipt"], receipt)
        staged_target = staged_paths["target"]
        _seal_fresh_recovery_tree_permissions(staged_target)
        staged_target_status = staged_target.stat(follow_symlinks=False)
        if not stat.S_ISDIR(staged_target_status.st_mode):
            raise MaterializationError("staged run-v17 root is invalid")
        verified_stage_identity = (
            staged_target_status.st_dev,
            staged_target_status.st_ino,
        )
        staged_verification = verify_fresh_generation_recovery(
            config,
            population,
            root=stage_root,
            source_root=authority_root,
            runtime_authority_root=root,
        )
        if (
            staged_verification.get("receipt_cid") != receipt["receipt_cid"]
            or staged_verification.get("operational_verification_root")
            != receipt["operational_verification_root"]
        ):
            raise MaterializationError("staged fresh recovery failed strict replay")
        if not staged_target.is_dir() or staged_target.is_symlink():
            raise MaterializationError("staged run-v17 root is invalid")
        verified_fsync_identity, verified_stage_fingerprint = _fsync_tree(
            staged_target
        )
        if verified_fsync_identity != verified_stage_identity:
            raise MaterializationError(
                "fresh recovery stage changed after strict verification"
            )
        if fault_injector is not None:
            fault_injector("after_stage_verification")
        if (
            _require_clean_recovery_source(
                config, population, source_root=authority_root
            )
            != source_binding
        ):
            raise MaterializationError(
                "fresh recovery source changed before atomic publish"
            )
        current_preview = preview_fresh_generation_recovery(
            config,
            population,
            root=root,
            source_root=authority_root,
        )
        if (
            _source_evidence_cid(current_preview)
            != manifest["source_evidence_cid"]
        ):
            raise MaterializationError(
                "fresh recovery forensic evidence changed before atomic publish"
            )
        observed_target, observed_state = _fresh_recovery_target_state(
            config, root=root
        )
        parent_status = os.fstat(parent_descriptor)
        lexical_parent_status = target.parent.stat(follow_symlinks=False)
        if (
            observed_target != target
            or observed_state != "absent"
            or (parent_status.st_dev, parent_status.st_ino)
            != (lexical_parent_status.st_dev, lexical_parent_status.st_ino)
        ):
            raise MaterializationError("run-v17 collision detected before atomic publish")
        staged_identity, staged_fingerprint = _fsync_tree(staged_target)
        if (
            staged_identity != verified_stage_identity
            or staged_fingerprint != verified_stage_fingerprint
        ):
            raise MaterializationError(
                "fresh recovery stage changed after strict verification"
            )
        _rename_noreplace(
            staged_target,
            parent_descriptor,
            target.name,
            expected_source_identity=verified_stage_identity,
        )
        os.fsync(parent_descriptor)
        if fault_injector is not None:
            fault_injector("after_publish")
        verification = verify_fresh_generation_recovery(
            config, population, root=root, source_root=authority_root
        )
        if verification.get("receipt_cid") != receipt.get("receipt_cid"):
            raise MaterializationError("published recovery receipt failed replay")
        return receipt
    finally:
        try:
            if stage_root is not None and stage_root.exists():
                shutil.rmtree(stage_root)
        finally:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)
            os.close(staging_descriptor)
            os.close(parent_descriptor)


def _materialize_canonical(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    root: Path = ROOT,
    recheck_source: bool = True,
) -> dict[str, Any]:
    """Install and populate one fresh embedded operational control plane."""

    paths = _paths(config, root=root)
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        names = ", ".join(path.relative_to(root).as_posix() for path in existing)
        raise MaterializationError(f"refusing to overwrite an existing control plane: {names}")
    if recheck_source:
        current = verify_source_binding(config, root=root)
        if (
            current["accelerator_head"] != population.get("source_head")
            or "git-tree:" + current["accelerator_tree"] != population.get("repository_tree_id")
            or current["source_forest_root"] != population.get("source_forest_root")
        ):
            raise MaterializationError("source forest changed after population construction")

    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        install_datasets_authoritative_operational_schema,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DatabaseImplementationDaemon,
    )

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        duckdb_version = importlib.metadata.version("duckdb")
    except importlib.metadata.PackageNotFoundError:
        duckdb_version = "unavailable"
    schema_install = install_datasets_authoritative_operational_schema(
        paths["control"],
        application_version="lgcvf-v1",
        tool_version=duckdb_version,
        owner_id="lgcvf-materializer:operational-schema",
    )
    prior_revision = os.environ.get(SCHEMA_REVISION_ENV)
    os.environ[SCHEMA_REVISION_ENV] = EXPECTED_SCHEMA_REVISION
    daemon: DatabaseImplementationDaemon | None = None
    try:
        daemon = DatabaseImplementationDaemon(
            database_path=paths["control"],
            coordination_path=paths["coordination"],
            execution_path=paths["execution"],
            owner_session_id="lgcvf-materializer:single-writer",
            authority_mode="embedded",
            task_source_kind="duckdb",
            install_schema=False,
        )
        database_receipt = daemon.materialize_population(
            population,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
        )
    finally:
        if daemon is not None:
            daemon.close()
        if prior_revision is None:
            os.environ.pop(SCHEMA_REVISION_ENV, None)
        else:
            os.environ[SCHEMA_REVISION_ENV] = prior_revision

    task_source_receipt = database_receipt.get("task_source")
    expected_task_cids = [str(item["task_cid"]) for item in population["tasks"]]
    if not isinstance(task_source_receipt, Mapping):
        raise MaterializationError("DatabaseTaskSource did not return a typed receipt")
    expected_counts = {
        "task_count": len(expected_task_cids),
        "goal_count": len(population["objectives"]),
        "goal_edge_count": len(population["goal_edges"]),
        "plan_count": 1,
        "task_cids": expected_task_cids,
    }
    if any(task_source_receipt.get(key) != value for key, value in expected_counts.items()):
        raise MaterializationError("DatabaseTaskSource receipt differs from the exact population")
    if list(database_receipt.get("registered_task_cids") or ()) != expected_task_cids:
        raise MaterializationError("DatabaseImplementationDaemon registration differs")
    expected_completed_cids = [
        str(item["task_cid"])
        for item in population["tasks"]
        if str(item.get("status") or "").strip().lower()
        in {"completed", "complete", "done"}
    ]
    if (
        list(database_receipt.get("bootstrap_completed_task_cids") or ())
        != expected_completed_cids
    ):
        raise MaterializationError(
            "DatabaseImplementationDaemon completion projection differs"
        )
    verification = _verify_read_only_core(
        config,
        population,
        root=root,
        expected_stage="initial",
    )
    receipt = {
        "schema": SCHEMA,
        "authority_mode": "embedded",
        "task_source_kind": "duckdb",
        "maximum_writer_processes": 1,
        "quack_qualified": False,
        "schema_revision": EXPECTED_SCHEMA_REVISION,
        "schema_profile": EXPECTED_SCHEMA_PROFILE,
        "semantic_truth_authority": "ipfs_datasets_py",
        "operational_coordination_authority": "ipfs_accelerate_py",
        "population_root": population["population_root"],
        "plan_root_cid": population["plan_root_cid"],
        "repository_tree_id": population["repository_tree_id"],
        "source_head": population["source_head"],
        "database_paths": {
            key: path.relative_to(root).as_posix()
            for key, path in sorted(paths.items())
            if key != "receipt"
        },
        "schema_install": (
            schema_install.to_dict()
            if callable(getattr(schema_install, "to_dict", None))
            else dict(schema_install)
        ),
        "materialization": dict(database_receipt),
        "verification": verification,
    }
    receipt["receipt_cid"] = content_identity(receipt)
    paths["receipt"].write_bytes(_canonical_bytes(receipt) + b"\n")
    return receipt


def materialize(
    config: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    root: Path = ROOT,
    recheck_source: bool = True,
) -> dict[str, Any]:
    """Install an ordinary control plane through the public materializer."""

    if _targets_fresh_recovery_generation(config, root=root):
        raise MaterializationError(
            "fresh recovery targets require recovery-materialize; canonical-only "
            "materialization is not admissible"
        )
    return _materialize_canonical(
        config, population, root=root, recheck_source=recheck_source
    )


def _load_receipt(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise MaterializationError("materialization receipt is absent or unreadable") from exc
    receipt = _strict_json_loads(raw, noun="materialization receipt")
    if not isinstance(receipt, dict):
        raise MaterializationError("materialization receipt must be an object")
    if raw != _canonical_bytes(receipt) + b"\n":
        raise MaterializationError("materialization receipt bytes are not canonical")
    claimed = str(receipt.pop("receipt_cid", ""))
    observed = content_identity(receipt)
    receipt["receipt_cid"] = claimed
    if not claimed or claimed != observed:
        raise MaterializationError("materialization receipt content identity does not verify")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "population",
            "materialize",
            "verify",
            "successor-preview",
            "successor-steer",
            "successor-verify",
            "recovery-preview",
            "recovery-materialize",
            "recovery-verify",
        ),
        help=(
            "render/bootstrap/verify the population, or preview, atomically steer, "
            "and read-only verify the immutable revision-2 continuation or fresh "
            "run-v17 evidence recovery"
        ),
    )
    args = parser.parse_args(argv)
    try:
        if args.command in {
            "materialize",
            "verify",
            "recovery-preview",
            "recovery-materialize",
            "recovery-verify",
        }:
            _require_isolated_recovery_interpreter()
        config = load_config()
        population = build_population(config)
        if args.command == "population":
            result: Mapping[str, Any] = population
        elif args.command == "materialize":
            raise MaterializationError(
                "run-v17 requires recovery-materialize; canonical bootstrap alone "
                "would omit six independently proved completions"
            )
        elif args.command == "successor-preview":
            result = preview_successor(config, population)
        elif args.command == "successor-steer":
            result = steer_successor(config, population)
        elif args.command == "successor-verify":
            result = verify_successor_read_only(config, population)
        elif args.command == "recovery-preview":
            result = preview_fresh_generation_recovery(config, population)
        elif args.command == "recovery-materialize":
            result = materialize_fresh_generation_recovery(config, population)
        elif args.command in {"recovery-verify", "verify"}:
            result = verify_fresh_generation_recovery(config, population)
        else:
            paths = _paths(config, root=ROOT)
            receipt = _load_receipt(paths["receipt"])
            if receipt.get("population_root") != population["population_root"]:
                raise MaterializationError("materialization receipt is stale for this population")
            result = verify_read_only(config, population, expected_stage="live")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except MaterializationError as exc:
        print(
            json.dumps(
                {"schema": SCHEMA, "valid": False, "error": str(exc)},
                indent=2,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
