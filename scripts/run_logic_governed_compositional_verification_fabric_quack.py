#!/usr/bin/env python3
"""Run the additive LGCVF DuckDB + Quack successor controller.

The canonical run-v16 database is forensic input and the sealed run-v17
configuration remains an embedded, single-writer recovery target.  This
operator therefore has two explicit stages:

* ``bootstrap`` verifies the canonical run-v17 recovery and publishes one
  no-overwrite run-v18 database clone with a provenance receipt;
* ``bootstrap-sealed-continuity`` admits a separately preserved run-v17 only
  through six explicit raw-byte pins, exact target-state reconstruction, and
  an operational-continuity-only authority ceiling;
* ``launch`` owns that clone in-process, starts exactly one foreground
  configured-board scheduler child, and services the closed mutation inbox.

The Quack attach credential exists only in the controller's memory and in the
trusted scheduler process environment.  It is never placed in argv, status,
logs, or a token-vault file.  Implementation-provider environments are still
scrubbed by the existing multi-supervisor boundary.

DuckLake is deliberately a separate, stopped-checkpoint observation.  The
``projection-once`` command writes a physically distinct BoardControlPlane
catalog and marks it non-authoritative; neither ``launch`` nor the configured
scheduler reads that projection for scheduling, leasing, or completion.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROGRAM_ROOT_RELATIVE: Final = Path(
    "data/agent_supervisor/logic_governed_compositional_verification_fabric"
)
SOURCE_RUN_RELATIVE: Final = PROGRAM_ROOT_RELATIVE / "run-v17"
SUCCESSOR_RUN_RELATIVE: Final = PROGRAM_ROOT_RELATIVE / "run-v18"
SOURCE_DATABASE_RELATIVE: Final = SOURCE_RUN_RELATIVE / "control.duckdb"
SUCCESSOR_DATABASE_RELATIVE: Final = SUCCESSOR_RUN_RELATIVE / "control.duckdb"
OWNER_STATE_RELATIVE: Final = SUCCESSOR_RUN_RELATIVE / "quack-owner"
PROVENANCE_RELATIVE: Final = (
    SUCCESSOR_RUN_RELATIVE / "evidence" / "quack-successor-provenance.json"
)
CONTROLLER_STATUS_RELATIVE: Final = SUCCESSOR_RUN_RELATIVE / "controller.status.json"
CONTROLLER_LOCK_RELATIVE: Final = SUCCESSOR_RUN_RELATIVE / "controller.lock"
CONTROLLER_LOG_RELATIVE: Final = SUCCESSOR_RUN_RELATIVE / "logs" / "scheduler.log"
PROJECTION_ROOT_RELATIVE: Final = SUCCESSOR_RUN_RELATIVE / "ducklake-board-projection"
PROJECTION_RECEIPT_RELATIVE: Final = (
    SUCCESSOR_RUN_RELATIVE / "evidence" / "ducklake-board-projection.json"
)
MATERIALIZER_RELATIVE: Final = Path(
    "scripts/materialize_logic_governed_compositional_verification_fabric_control_plane.py"
)
DEFAULT_SUCCESSOR_CONFIG_RELATIVE: Final = Path(
    "config/agent_supervisor_logic_governed_compositional_verification_fabric_quack_candidate_scheduler.json"
)

PROVENANCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-quack-successor-provenance@2"
)
SEALED_CONTINUITY_VERIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "lgcvf-target-only-initial-continuity-verification@1"
)
SEALED_CONTINUITY_MODE: Final = "target_only_initial_continuity"
SEALED_CONTINUITY_AUTHORITY_CEILING: Final = "operational_continuity_only"
FRESH_RECOVERY_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-fresh-generation-recovery-receipt@1"
)
FRESH_RECOVERY_MANIFEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-fresh-generation-recovery-manifest@1"
)
BOOTSTRAP_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-duckdb-materialization@1"
)
CONTROLLER_STATUS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-quack-successor-status@1"
)
PROJECTION_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/lgcvf-ducklake-board-projection@1"
)
TOKEN_ENV: Final = "IPFS_ACCELERATE_AGENT_QUACK_TOKEN"
TOKEN_FILE_ENV: Final = "IPFS_ACCELERATE_AGENT_QUACK_TOKEN_FILE"
STORE_GENERATION_ENV: Final = "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION"
BOARD_EXTENSION_INSTALL_POLICY_ENV: Final = (
    "IPFS_ACCELERATE_AGENT_BOARD_EXTENSION_INSTALL_POLICY"
)
BOARD_EXTENSION_INSTALL_POLICY_LOAD_ONLY: Final = "load_only"
SECRET_HANDLE: Final = f"env://{TOKEN_ENV}"
APPROVED_BOARD_BRANCH: Final = (
    "agent/logic-governed-compositional-verification-fabric-v1"
)
APPROVED_REMOTE_BRANCH_REF: Final = "refs/remotes/github/" + APPROVED_BOARD_BRANCH
MAX_DATABASE_BYTES: Final = 8 * 1024 * 1024 * 1024
MAX_JSON_BYTES: Final = 4 * 1024 * 1024
MAX_SECRET_SURFACE_BYTES: Final = 1024 * 1024 * 1024
MAX_STOP_SECONDS: Final = 20.0
UNIX_SOCKET_PATH_CEILING: Final = 100
COMPLETED_TASK_IDS: Final = (
    "LGCVF-001",
    "LGCVF-002",
    "LGCVF-010",
    "LGCVF-020",
    "LGCVF-030",
    "LGCVF-040",
    "LGCVF-050",
    "LGCVF-051",
    "LGCVF-060",
    "LGCVF-061",
    "LGCVF-070",
    "LGCVF-071",
    "LGCVF-080",
)
TODO_TASK_IDS: Final = (
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
BLOCKED_TASK_IDS: Final = ("LGCVF-121", "LGCVF-123")
CONSTRUCTION_COMPLETED_TASK_IDS: Final = COMPLETED_TASK_IDS[:7]
RECOVERED_COMPLETED_TASK_IDS: Final = COMPLETED_TASK_IDS[7:]
SEALED_CONTINUITY_EXPECTED_PINS: Final = {
    "control_sha256": (
        "sha256:c931eb71c8ef861c0b4823341989298311a11414b5a7e69ec13f74db62c09238"
    ),
    "coordination_sha256": (
        "sha256:1882695aba63a3d872cbbb6bb737eb173ea81fd9e0b8b6a5131f11f10f7fa2c4"
    ),
    "execution_sha256": (
        "sha256:ca13093d54c55461eea9250b36a06f16764b51e70f0e25965efb207bafd7e9a5"
    ),
    "bootstrap_sha256": (
        "sha256:dd8baaeaf285a23a4e848f03e4a1fd0532c4127e67210d63896e557219b126ab"
    ),
    "manifest_sha256": (
        "sha256:ba418511fec39660765763b012781b8109d437dc02008c01aa1374f843727c71"
    ),
    "recovery_receipt_sha256": (
        "sha256:24fcad13eb74537b1cd0f7531e27282833a77782323aba4a9e2b98c787b013f2"
    ),
}
SEALED_CONTINUITY_EXPECTED_IDENTITIES: Final = {
    "bootstrap_receipt_cid": (
        "baguqeeraujtyr6ywjlmjagd5ijtvkvcxkag5hrtdhyonb66cyhfq55zpfvaa"
    ),
    "manifest_cid": ("baguqeeravix5cxsnflvjmvniwzpqtkrstappy3z5vgjehgk2xlwdn3yhq62a"),
    "receipt_cid": ("baguqeeramzbpvvpb262jwlqa627d4zbqip6tlg6q5gxycdvr4gaoqonpt5ca"),
    "population_root": (
        "baguqeerar2vrvf44pbumffg65zh5etmged3va3ocumu75v3fdgzqbzlk4nja"
    ),
    "source_evidence_cid": (
        "baguqeera4aybmwbobzlojc4u2cdqxxznmd4bgjkwv2kqka5cukhywnmhy4uq"
    ),
    "sealed_operational_verification_root": (
        "baguqeeraqdjtxgx6wjxkb6u3635s633xy7ymqjby4xxo7xq6wrstfnzym4pa"
    ),
    "target_source_head": "092c95725b9642daa479162d631eff3983e67af6",
    "target_source_tree": "83488b19d20f06da44762a2dfecb4a2666c3b192",
}
GIT_EXECUTABLE: Final = Path("/usr/bin/git")
GIT_TIMEOUT_SECONDS: Final = 120.0


class SuccessorOperatorError(RuntimeError):
    """The successor cannot be admitted without weakening a boundary."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: Any) -> str:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )

    return content_identity(value)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _contained(root: Path, relative: Path | str) -> Path:
    base = root.resolve()
    candidate = (base / Path(relative)).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise SuccessorOperatorError(
            f"runtime path escapes repository: {relative}"
        ) from exc
    return candidate


def _paths(root: Path = ROOT) -> dict[str, Path]:
    paths = {
        "source_database": _contained(root, SOURCE_DATABASE_RELATIVE),
        "successor_database": _contained(root, SUCCESSOR_DATABASE_RELATIVE),
        "owner_state": _contained(root, OWNER_STATE_RELATIVE),
        "provenance": _contained(root, PROVENANCE_RELATIVE),
        "controller_status": _contained(root, CONTROLLER_STATUS_RELATIVE),
        "controller_lock": _contained(root, CONTROLLER_LOCK_RELATIVE),
        "controller_log": _contained(root, CONTROLLER_LOG_RELATIVE),
        "projection_root": _contained(root, PROJECTION_ROOT_RELATIVE),
        "projection_receipt": _contained(root, PROJECTION_RECEIPT_RELATIVE),
    }
    socket_identity = hashlib.sha256(
        _canonical_bytes(
            {
                "program": "lgcvf-quack-successor-v1",
                "repository_root": str(root.resolve()),
                "runtime_root": str(_contained(root, SUCCESSOR_RUN_RELATIVE)),
                "database": str(paths["successor_database"]),
            }
        )
    ).hexdigest()[:20]
    owner_socket = (
        Path(tempfile.gettempdir())
        / f"ipfs-accelerate-lgcvf-{os.geteuid()}"
        / f"owner-{socket_identity}.sock"
    )
    if len(os.fsencode(owner_socket)) > UNIX_SOCKET_PATH_CEILING:
        raise SuccessorOperatorError(
            "derived state-owner socket path exceeds its bound"
        )
    paths["owner_socket"] = owner_socket
    return paths


def _read_bounded_regular_file(
    path: Path,
    *,
    max_bytes: int,
    noun: str,
    require_private_owner: bool = False,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SuccessorOperatorError(f"{noun} is unreadable: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 0
            or before.st_size > max_bytes
            or (
                require_private_owner
                and (
                    before.st_uid != os.geteuid()
                    or before.st_nlink != 1
                    or stat.S_IMODE(before.st_mode) & 0o077
                )
            )
        ):
            raise SuccessorOperatorError(
                f"{noun} is not a bounded private regular file: {path}"
            )
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(raw) > max_bytes or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise SuccessorOperatorError(f"{noun} changed while reading: {path}")
        return raw
    finally:
        os.close(descriptor)


def _strict_json(
    path: Path,
    *,
    expected_schema: str = "",
    require_private_owner: bool = False,
) -> dict[str, Any]:
    raw = _read_bounded_regular_file(
        path,
        max_bytes=MAX_JSON_BYTES,
        noun="required receipt",
        require_private_owner=require_private_owner,
    )
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SuccessorOperatorError(f"receipt is malformed: {path}") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
        raise SuccessorOperatorError(f"receipt is not a canonical object: {path}")
    if expected_schema and value.get("schema") != expected_schema:
        raise SuccessorOperatorError(f"receipt schema differs: {path}")
    claimed = str(value.get("receipt_cid") or value.get("status_cid") or "")
    if claimed:
        unsigned = dict(value)
        unsigned.pop("receipt_cid", None)
        unsigned.pop("status_cid", None)
        if claimed != _content_id(unsigned):
            raise SuccessorOperatorError(f"receipt content identity differs: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any], *, replace: bool) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(dict(value)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise SuccessorOperatorError(f"refusing to overwrite {path}") from exc
            temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _rename_directory_noreplace(
    parent_descriptor: int, source_name: str, target_name: str
) -> None:
    """Atomically publish one same-parent directory without an overwrite fallback."""

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise SuccessorOperatorError(
            "atomic no-replace directory publication is unavailable"
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(target_name),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    observed_errno = ctypes.get_errno()
    if observed_errno in (errno.EEXIST, errno.ENOTEMPTY):
        raise SuccessorOperatorError("refusing to overwrite an existing successor")
    raise SuccessorOperatorError(
        "atomic no-replace successor publication failed: " + os.strerror(observed_errno)
    )


def _cleanup_successor_stage(
    stage: Path, *, staged_database: Path, staged_provenance: Path
) -> None:
    """Remove only the exact unpublished objects this process created."""

    lock_paths = tuple(
        stage / name
        for name in (
            f".{staged_database.name}.intent.lock",
            f".{staged_database.name}.lock",
            f".{staged_database.name}.migration.lock",
        )
    )
    for path in (staged_provenance, staged_database, *lock_paths):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    cursor = staged_provenance.parent
    while cursor != stage:
        try:
            cursor.rmdir()
        except (FileNotFoundError, OSError):
            break
        cursor = cursor.parent
    try:
        stage.rmdir()
    except (FileNotFoundError, OSError):
        pass


def _remove_staged_database_locks(stage: Path, database_name: str) -> None:
    """Remove only empty, owner-held lock artifacts created by read verification."""

    for name in (
        f".{database_name}.intent.lock",
        f".{database_name}.lock",
        f".{database_name}.migration.lock",
    ):
        path = stage / name
        try:
            metadata = os.lstat(path)
        except FileNotFoundError:
            continue
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_size != 0
        ):
            raise SuccessorOperatorError("staged database lock custody differs")
        path.unlink()


def _open_private_lock(path: Path) -> Any:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
        ):
            raise SuccessorOperatorError("controller lock custody is unsafe")
        os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "a+b")
    except BaseException:
        os.close(descriptor)
        raise


def _sha256_regular_file(
    path: Path,
    *,
    max_bytes: int = MAX_DATABASE_BYTES,
    noun: str = "database",
    require_private_owner: bool = False,
) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SuccessorOperatorError(f"{noun} is unreadable: {path}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > max_bytes
            or (
                require_private_owner
                and (
                    before.st_uid != os.geteuid()
                    or before.st_nlink != 1
                    or stat.S_IMODE(before.st_mode) & 0o077
                )
            )
        ):
            raise SuccessorOperatorError(
                f"{noun} is not a bounded private regular file: {path}"
            )
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise SuccessorOperatorError(f"{noun} changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return "sha256:" + digest.hexdigest()


def _regular_file_contains(path: Path, needle: bytes) -> bool:
    if not needle:
        return False
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise SuccessorOperatorError(
            f"could not inspect credential surface: {path}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size < 0
            or metadata.st_size > MAX_SECRET_SURFACE_BYTES
        ):
            raise SuccessorOperatorError(
                f"credential surface is not a bounded regular file: {path}"
            )
        carry = b""
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                return False
            observed = carry + block
            if needle in observed:
                return True
            overlap = max(0, len(needle) - 1)
            carry = observed[-overlap:] if overlap else b""
    finally:
        os.close(descriptor)


def _database_identity(path: Path) -> dict[str, str]:
    import duckdb

    try:
        connection = duckdb.connect(str(path), read_only=True)
        try:
            rows = connection.execute(
                "SELECT key, value FROM control_plane_metadata "
                "WHERE key IN ('database_uuid','schema_version',"
                "'schema_fingerprint','migration_catalog_fingerprint')"
            ).fetchall()
        finally:
            connection.close()
    except Exception as exc:
        raise SuccessorOperatorError(
            f"could not read control-plane identity from {path}: {type(exc).__name__}"
        ) from exc
    return {str(key): str(value or "") for key, value in rows}


def datasets_profile_migration(path: Path) -> Any:
    """Idempotently admit the datasets-authoritative migration catalog."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        install_datasets_authoritative_operational_schema,
        load_datasets_authoritative_operational_catalog,
        verify_datasets_authoritative_operational_schema,
    )

    report = install_datasets_authoritative_operational_schema(
        path,
        application_version="lgcvf-quack-successor-v1",
        tool_version="lgcvf-quack-controller-v1",
        owner_id=f"lgcvf-quack-controller:{os.getpid()}",
    )
    verification = verify_datasets_authoritative_operational_schema(path)
    expected_catalog = load_datasets_authoritative_operational_catalog().fingerprint()
    if (
        verification.get("valid") is not True
        or report.schema_fingerprint != verification.get("schema_fingerprint")
        or report.catalog_fingerprint != expected_catalog
        or verification.get("catalog_fingerprint") != expected_catalog
    ):
        raise SuccessorOperatorError(
            "datasets-authoritative migration report and verification differ"
        )
    return report


def _verify_profile(path: Path) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        load_datasets_authoritative_operational_catalog,
        verify_datasets_authoritative_operational_schema,
    )

    verification = verify_datasets_authoritative_operational_schema(path)
    expected = load_datasets_authoritative_operational_catalog().fingerprint()
    if (
        verification.get("valid") is not True
        or verification.get("catalog_fingerprint") != expected
    ):
        raise SuccessorOperatorError(
            f"datasets-authoritative schema verification failed: {path}"
        )
    return verification


def _strict_addressed_mapping(
    value: Mapping[str, Any],
    *,
    identity_field: str,
    noun: str,
) -> dict[str, Any]:
    normalized = dict(value)
    claimed = str(normalized.get(identity_field) or "")
    unsigned = dict(normalized)
    unsigned.pop(identity_field, None)
    if not claimed or claimed != _content_id(unsigned):
        raise SuccessorOperatorError(f"{noun} content identity differs")
    return normalized


def _strict_addressed_json(
    path: Path,
    *,
    expected_schema: str,
    identity_field: str,
    noun: str,
) -> dict[str, Any]:
    value = _strict_json(
        path,
        expected_schema=expected_schema,
        require_private_owner=True,
    )
    return _strict_addressed_mapping(
        value,
        identity_field=identity_field,
        noun=noun,
    )


def _plain_json_object(path: Path, *, noun: str) -> dict[str, Any]:
    raw = _read_bounded_regular_file(path, max_bytes=MAX_JSON_BYTES, noun=noun)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SuccessorOperatorError(f"{noun} is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise SuccessorOperatorError(f"{noun} is not an object: {path}")
    return value


def _require_sha256_pin(value: str, *, noun: str) -> str:
    normalized = str(value or "")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", normalized) is None:
        raise SuccessorOperatorError(f"{noun} SHA-256 pin is malformed")
    return normalized


def _require_private_directory(path: Path, *, noun: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise SuccessorOperatorError(f"{noun} is unavailable: {path}") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise SuccessorOperatorError(f"{noun} custody is not private: {path}")


def _privatize_owned_directory(path: Path, *, noun: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise SuccessorOperatorError(f"{noun} is unavailable: {path}") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
    ):
        raise SuccessorOperatorError(f"{noun} is not an owned directory: {path}")
    os.chmod(path, 0o700, follow_symlinks=False)
    _require_private_directory(path, noun=noun)


def _sealed_source_paths(source_root: Path) -> dict[str, Path]:
    lexical = Path(os.path.abspath(os.fspath(source_root)))
    if lexical.name != "run-v17":
        raise SuccessorOperatorError("sealed continuity source must be named run-v17")
    cursor = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        cursor = cursor / component
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise SuccessorOperatorError(
                "sealed continuity source path cannot be inspected"
            ) from exc
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise SuccessorOperatorError(
                "sealed continuity source path contains a link or non-directory"
            )
    _require_private_directory(lexical, noun="sealed continuity source")
    evidence = lexical / "evidence"
    bootstrap_root = evidence / "bootstrap"
    recovery_root = evidence / "fresh-generation-recovery"
    for directory, noun in (
        (evidence, "sealed evidence directory"),
        (bootstrap_root, "sealed bootstrap directory"),
        (recovery_root, "sealed recovery directory"),
    ):
        _require_private_directory(directory, noun=noun)
    paths = {
        "root": lexical,
        "control": lexical / "control.duckdb",
        "coordination": lexical / "control.coordination.duckdb",
        "execution": lexical / "control.execution.duckdb",
        "bootstrap": bootstrap_root / "materialization.json",
        "recovery_root": recovery_root,
        "recovery_receipt": recovery_root / "recovery-receipt.json",
    }
    for key in ("control", "coordination", "execution"):
        if paths[key].with_name(paths[key].name + ".wal").exists():
            raise SuccessorOperatorError(f"sealed {key} database has a live WAL")
    return paths


def _git_text(root: Path, arguments: Sequence[str], *, noun: str) -> str:
    environment = {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
    }
    completed = subprocess.run(
        [
            str(GIT_EXECUTABLE),
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            *arguments,
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=GIT_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise SuccessorOperatorError(
            f"{noun} failed: {(completed.stderr or completed.stdout)[-1000:].strip()}"
        )
    return completed.stdout.strip()


def _git_quiet(root: Path, arguments: Sequence[str], *, noun: str) -> None:
    _git_text(root, arguments, noun=noun)


def _target_source_continuity(
    root: Path,
    *,
    source_head: str,
    source_tree: str,
    config: Mapping[str, Any],
) -> dict[str, str]:
    if (
        re.fullmatch(r"[0-9a-f]{40}", source_head) is None
        or re.fullmatch(r"[0-9a-f]{40}", source_tree) is None
    ):
        raise SuccessorOperatorError("sealed source Git identity is malformed")
    branch = _git_text(root, ("symbolic-ref", "--short", "HEAD"), noun="board branch")
    if branch != APPROVED_BOARD_BRANCH or config.get("merge_target_branch") != branch:
        raise SuccessorOperatorError(
            "continuity verification is not on the approved board branch"
        )
    current_head = _git_text(root, ("rev-parse", "HEAD"), noun="current HEAD")
    current_tree = _git_text(root, ("rev-parse", "HEAD^{tree}"), noun="current tree")
    dirty = _git_text(
        root,
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ),
        noun="candidate source inventory",
    )
    if dirty:
        raise SuccessorOperatorError(
            "continuity verification requires a completely clean candidate worktree"
        )
    datasets_relative = "ipfs_datasets_py"
    datasets = _contained(root, datasets_relative)
    datasets_metadata = os.lstat(datasets)
    if (
        not stat.S_ISDIR(datasets_metadata.st_mode)
        or stat.S_ISLNK(datasets_metadata.st_mode)
        or datasets_metadata.st_uid != os.geteuid()
    ):
        raise SuccessorOperatorError("nested runtime source custody differs")
    datasets_head = _git_text(
        datasets, ("rev-parse", "HEAD"), noun="nested runtime HEAD"
    )
    datasets_tree = _git_text(
        datasets, ("rev-parse", "HEAD^{tree}"), noun="nested runtime tree"
    )
    datasets_dirty = _git_text(
        datasets,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        noun="nested runtime source inventory",
    )
    gitlink = _git_text(
        root,
        ("ls-tree", current_head, "--", datasets_relative),
        noun="nested runtime gitlink",
    ).split()
    if (
        datasets_dirty
        or len(gitlink) < 3
        or gitlink[0] != "160000"
        or gitlink[1] != "commit"
        or gitlink[2] != datasets_head
    ):
        raise SuccessorOperatorError(
            "continuity verification requires the exact clean nested runtime gitlink"
        )
    remote_head = _git_text(
        root,
        ("rev-parse", APPROVED_REMOTE_BRANCH_REF),
        noun="resolved remote board branch",
    )
    if current_head != remote_head:
        raise SuccessorOperatorError(
            "current board candidate is not the resolved remote branch"
        )
    observed_source_tree = _git_text(
        root,
        ("show", "-s", "--format=%T", source_head),
        noun="sealed source commit",
    )
    if observed_source_tree != source_tree:
        raise SuccessorOperatorError("sealed source commit/tree binding differs")
    _git_quiet(
        root,
        ("merge-base", "--is-ancestor", source_head, current_head),
        noun="sealed source ancestry",
    )
    authority_paths = []
    for field in (
        "taskboard_path",
        "objectives_path",
        "plan_path",
        "formal_plan_path",
        "validator_path",
    ):
        value = str(config.get(field) or "")
        if not value or Path(value).is_absolute() or ".." in Path(value).parts:
            raise SuccessorOperatorError(f"scheduler {field} is unsafe")
        authority_paths.append(value)
    config_relative = (
        "config/agent_supervisor_logic_governed_compositional_verification_fabric_"
        "scheduler.json"
    )
    _git_quiet(
        root,
        (
            "diff",
            "--no-ext-diff",
            "--quiet",
            "HEAD",
            "--",
            config_relative,
            *authority_paths,
        ),
        noun="current authority source worktree",
    )
    _git_quiet(
        root,
        (
            "diff",
            "--no-ext-diff",
            "--quiet",
            source_head,
            current_head,
            "--",
            *authority_paths,
        ),
        noun="sealed/current authority source",
    )
    return {
        "approved_branch": branch,
        "resolved_remote_head": remote_head,
        "current_head": current_head,
        "current_tree": current_tree,
        "candidate_worktree_clean": "true",
        "datasets_head": datasets_head,
        "datasets_tree": datasets_tree,
        "datasets_worktree_clean": "true",
        "target_source_head": source_head,
        "target_source_tree": source_tree,
    }


def _require_false_authority(value: Mapping[str, Any], *, noun: str) -> None:
    false_fields = (
        "validation_self_authority",
        "validation_completion_authoritative",
        "task_implementation_complete",
        "test_qualification_complete",
        "objective_complete",
        "release_qualified",
        "production_authorized",
    )
    if any(value.get(field) is not False for field in false_fields):
        raise SuccessorOperatorError(f"{noun} exceeds the continuity authority ceiling")
    if (
        value.get("candidate_authored_validation") is not True
        or value.get("network_isolation_enforced") is not True
        or value.get("model_provider_route") != "none"
        or value.get("source_database_statuses_read") is not False
        or value.get("source_database_completion_records_imported") is not False
        or value.get("synthetic_source_disposition") != "quarantined_not_imported"
    ):
        raise SuccessorOperatorError(f"{noun} recovery limitations differ")


def _validate_recovery_policy_projection(
    *,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    policy = config.get("fresh_generation_recovery")
    plan_binding = config.get("plan_binding")
    if not isinstance(policy, Mapping) or not isinstance(plan_binding, Mapping):
        raise SuccessorOperatorError("tracked fresh-recovery policy is unavailable")
    expected_partition = {
        "construction_completed_task_ids": list(CONSTRUCTION_COMPLETED_TASK_IDS),
        "recovered_completed_task_ids": list(RECOVERED_COMPLETED_TASK_IDS),
        "rejected_synthetic_task_ids": list(TODO_TASK_IDS),
        "preserved_blocked_task_ids": list(BLOCKED_TASK_IDS),
        "completed_count": 13,
        "todo_count": 13,
        "blocked_count": 2,
    }
    if manifest.get("completion_partition") != expected_partition:
        raise SuccessorOperatorError("sealed completion partition differs")
    retained = manifest.get("retained_completion_binding")
    expected_retained = {
        "binding_cid": policy.get("retained_completion_binding_cid"),
        "construction_completion_count": 7,
        "delta_cid": policy.get("retained_delta_cid"),
        "dynamic_completion_receipt_count": 5,
        "logical_completion_count": 12,
        "path": policy.get("retained_revision_receipt_path"),
        "protected_blocker_binding_cid": policy.get(
            "retained_protected_blocker_binding_cid"
        ),
        "receipt_cid": policy.get("retained_revision_receipt_cid"),
        "sha256": policy.get("retained_revision_receipt_sha256"),
        "successor_revision_cid": policy.get("retained_successor_revision_cid"),
    }
    if retained != expected_retained:
        raise SuccessorOperatorError("sealed retained-completion projection differs")
    quarantine = manifest.get("wrong_default_quarantine")
    if not isinstance(quarantine, Mapping):
        raise SuccessorOperatorError("sealed wrong-default quarantine is unavailable")
    quarantine_projection = {
        "incident_manifest_path": policy.get("wrong_default_incident_manifest_path"),
        "incident_manifest_sha256": policy.get(
            "wrong_default_incident_manifest_sha256"
        ),
        "incident_manifest_cid": policy.get("wrong_default_incident_manifest_cid"),
        "contaminated_coordination_manifest_path": policy.get(
            "contaminated_coordination_projection_path"
        ),
        "contaminated_coordination_manifest_sha256": policy.get(
            "contaminated_coordination_projection_sha256"
        ),
        "contaminated_coordination_manifest_cid": policy.get(
            "contaminated_coordination_projection_manifest_cid"
        ),
        "rejected_record_set_cid": policy.get(
            "contaminated_coordination_rejected_record_set_cid"
        ),
        "rejected_contaminated_coordination_projection_root": policy.get(
            "rejected_contaminated_coordination_projection_root"
        ),
        "rejected_synthetic_task_ids": list(TODO_TASK_IDS),
        "disposition": "preserved_forensic_quarantine_not_imported",
        "source_database_opened": False,
    }
    if any(
        quarantine.get(key) != value for key, value in quarantine_projection.items()
    ):
        raise SuccessorOperatorError(
            "sealed wrong-default quarantine projection differs"
        )
    policy_merges = policy.get("merge_completions")
    manifest_merges = manifest.get("merge_completion_evidence")
    if (
        not isinstance(policy_merges, list)
        or not isinstance(manifest_merges, list)
        or len(policy_merges) != len(RECOVERED_COMPLETED_TASK_IDS)
        or len(manifest_merges) != len(policy_merges)
    ):
        raise SuccessorOperatorError("sealed merge-completion population differs")
    for expected, observed in zip(policy_merges, manifest_merges, strict=True):
        if (
            not isinstance(expected, Mapping)
            or not isinstance(observed, Mapping)
            or any(observed.get(key) != value for key, value in expected.items())
        ):
            raise SuccessorOperatorError("sealed merge-completion projection differs")
    common_fields = (
        "source_generation",
        "target_generation",
        "source_head",
        "source_tree",
        "source_evidence_cid",
        "plan_root_cid",
        "population_root",
        "validation_qualification_cid",
        "candidate_authored_validation",
        "validation_self_authority",
        "validation_completion_authoritative",
        "source_database_statuses_read",
        "source_database_completion_records_imported",
        "synthetic_source_disposition",
        "network_isolation_enforced",
        "model_provider_route",
        "task_implementation_complete",
        "test_qualification_complete",
        "objective_complete",
        "release_qualified",
        "production_authorized",
    )
    if any(manifest.get(field) != receipt.get(field) for field in common_fields):
        raise SuccessorOperatorError(
            "sealed recovery receipt/manifest projection differs"
        )
    if (
        manifest.get("source_generation") != policy.get("source_generation")
        or manifest.get("target_generation") != policy.get("target_generation")
        or manifest.get("source_runtime_root") != policy.get("source_runtime_root")
        or manifest.get("target_runtime_root") != policy.get("target_runtime_root")
        or manifest.get("plan_root_cid") != plan_binding.get("formal_plan_content_id")
        or receipt.get("completed_task_ids") != list(COMPLETED_TASK_IDS)
        or receipt.get("todo_task_ids") != list(TODO_TASK_IDS)
        or receipt.get("blocked_task_ids") != list(BLOCKED_TASK_IDS)
        or receipt.get("completed_count") != 13
        or receipt.get("todo_count") != 13
        or receipt.get("blocked_count") != 2
        or receipt.get("atomic_publish") is not True
    ):
        raise SuccessorOperatorError("sealed recovery policy binding differs")
    _require_false_authority(manifest, noun="sealed recovery manifest")
    _require_false_authority(receipt, noun="sealed recovery receipt")


def _validate_historical_qualification(manifest: Mapping[str, Any]) -> None:
    qualification = manifest.get("validation_qualification")
    if not isinstance(qualification, Mapping):
        raise SuccessorOperatorError("sealed historical qualification is unavailable")
    normalized = _strict_addressed_mapping(
        qualification,
        identity_field="receipt_cid",
        noun="sealed historical qualification",
    )
    if (
        normalized.get("receipt_cid") != manifest.get("validation_qualification_cid")
        or normalized.get("passed") is not True
        or normalized.get("disposition") != "passed"
        or normalized.get("candidate_authored_replay") is not True
        or normalized.get("completion_authoritative") is not False
        or normalized.get("production_authoritative") is not False
        or normalized.get("production_authorized") is not False
        or normalized.get("objective_complete") is not False
        or normalized.get("provider_route") != "none"
        or normalized.get("network_permitted") is not False
        or normalized.get("cache_reused") is not False
    ):
        raise SuccessorOperatorError(
            "sealed historical qualification limitations differ"
        )
    recovery_manifest = normalized.get("recovery_manifest")
    if not isinstance(recovery_manifest, Mapping):
        raise SuccessorOperatorError("historical qualification manifest is unavailable")
    _strict_addressed_mapping(
        recovery_manifest,
        identity_field="manifest_cid",
        noun="historical qualification manifest",
    )


def _verify_sealed_control_state(
    database: Path,
    *,
    expected_sha256: str,
    manifest: Mapping[str, Any],
    formal_plan: Mapping[str, Any],
) -> dict[str, Any]:
    import duckdb

    before = _sha256_regular_file(
        database,
        noun="sealed control database",
        require_private_owner=True,
    )
    if before != expected_sha256:
        raise SuccessorOperatorError("sealed control database SHA-256 differs")
    profile = _verify_profile(database)
    formal_tasks = formal_plan.get("tasks")
    if not isinstance(formal_tasks, list):
        raise SuccessorOperatorError("tracked formal task population is unavailable")
    formal_by_alias = {
        str(item.get("task_id") or ""): dict(item)
        for item in formal_tasks
        if isinstance(item, Mapping)
    }
    all_aliases = set(COMPLETED_TASK_IDS + TODO_TASK_IDS + BLOCKED_TASK_IDS)
    if set(formal_by_alias) != all_aliases:
        raise SuccessorOperatorError("tracked formal task population differs")
    try:
        connection = duckdb.connect(
            str(database),
            read_only=True,
            config={
                "autoinstall_known_extensions": "false",
                "autoload_known_extensions": "false",
            },
        )
        try:
            task_rows = connection.execute(
                "SELECT task_cid, task_alias, status, revision, plan_cid, "
                "identity_json, body_json FROM tasks ORDER BY task_alias"
            ).fetchall()
            plan_rows = connection.execute(
                "SELECT plan_cid, plan_alias, status, revision, body_json "
                "FROM plans ORDER BY plan_cid"
            ).fetchall()
            dependency_rows = connection.execute(
                "SELECT task_cid, dependency_task_cid, kind "
                "FROM task_dependencies ORDER BY task_cid, dependency_task_cid, kind"
            ).fetchall()
            completion_rows = connection.execute(
                "SELECT task_cid FROM completion_receipts ORDER BY task_cid"
            ).fetchall()
            zero_counts = {
                table: int(
                    connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                )
                for table in (
                    "task_claims",
                    "task_attempts",
                    "task_assignments",
                    "task_blocks",
                    "resource_claims",
                    "maintenance_leases",
                    "leases",
                    "lease_events",
                    "token_history",
                    "client_sessions",
                )
            }
        finally:
            connection.close()
    except Exception as exc:
        if isinstance(exc, SuccessorOperatorError):
            raise
        raise SuccessorOperatorError(
            f"sealed control state cannot be reconstructed: {type(exc).__name__}"
        ) from exc
    if len(task_rows) != 28 or any(zero_counts.values()):
        raise SuccessorOperatorError(
            "sealed control database has unexpected live state"
        )
    expected_status_revision = {
        **{alias: ("completed", 1) for alias in CONSTRUCTION_COMPLETED_TASK_IDS},
        **{alias: ("completed", 2) for alias in RECOVERED_COMPLETED_TASK_IDS},
        **{alias: ("todo", 1) for alias in TODO_TASK_IDS},
        **{alias: ("blocked", 1) for alias in BLOCKED_TASK_IDS},
    }
    tasks_by_cid: dict[str, str] = {}
    rows_by_alias: dict[str, dict[str, Any]] = {}
    for (
        task_cid,
        alias,
        status,
        revision,
        plan_cid,
        identity_raw,
        body_raw,
    ) in task_rows:
        task_cid = str(task_cid)
        alias = str(alias)
        try:
            identity = json.loads(str(identity_raw))
            body = json.loads(str(body_raw))
        except json.JSONDecodeError as exc:
            raise SuccessorOperatorError("sealed task JSON is malformed") from exc
        if (
            alias not in expected_status_revision
            or (str(status), int(revision)) != expected_status_revision[alias]
            or str(plan_cid) != manifest.get("plan_root_cid")
            or not isinstance(identity, Mapping)
            or identity.get("task_alias") != alias
            or identity.get("task_cid") != task_cid
            or identity.get("repository_tree_id")
            != "git-tree:" + str(manifest.get("source_tree") or "")
            or not isinstance(body, Mapping)
            or body.get("formal_record") != formal_by_alias[alias]
            or body.get("formal_task_content_id") != task_cid
            or body.get("board_namespace")
            != "logic-governed-compositional-verification-fabric-v1"
        ):
            raise SuccessorOperatorError(f"{alias}: sealed task authority differs")
        if task_cid in tasks_by_cid or alias in rows_by_alias:
            raise SuccessorOperatorError("sealed task identity is duplicated")
        tasks_by_cid[task_cid] = alias
        rows_by_alias[alias] = {
            "task_cid": task_cid,
            "status": str(status),
            "body": dict(body),
        }
    if set(rows_by_alias) != all_aliases:
        raise SuccessorOperatorError("sealed task alias population differs")
    if len(plan_rows) != 1:
        raise SuccessorOperatorError("sealed plan population differs")
    plan_cid, plan_alias, plan_status, plan_revision, plan_body_raw = plan_rows[0]
    try:
        plan_body = json.loads(str(plan_body_raw))
    except json.JSONDecodeError as exc:
        raise SuccessorOperatorError("sealed plan JSON is malformed") from exc
    if (
        str(plan_cid) != manifest.get("plan_root_cid")
        or str(plan_alias) != "logic-governed-compositional-verification-fabric-v1"
        or str(plan_status) != "active"
        or int(plan_revision) != 1
        or not isinstance(plan_body, Mapping)
        or plan_body.get("source_head") != manifest.get("source_head")
        or plan_body.get("repository_tree_id")
        != "git-tree:" + str(manifest.get("source_tree") or "")
    ):
        raise SuccessorOperatorError("sealed active plan differs")
    observed_dependencies: set[tuple[str, str]] = set()
    for task_cid, dependency_cid, kind in dependency_rows:
        task_alias = tasks_by_cid.get(str(task_cid), "")
        dependency_alias = tasks_by_cid.get(str(dependency_cid), "")
        if not task_alias or not dependency_alias or str(kind) != "depends_on":
            raise SuccessorOperatorError("sealed dependency identity differs")
        observed_dependencies.add((task_alias, dependency_alias))
    expected_dependencies = {
        (alias, str(dependency))
        for alias, task in formal_by_alias.items()
        for dependency in task.get("depends_on") or ()
    }
    if (
        len(dependency_rows) != 46
        or len(observed_dependencies) != 46
        or observed_dependencies != expected_dependencies
    ):
        raise SuccessorOperatorError("sealed dependency graph differs")
    completed_cids = {rows_by_alias[alias]["task_cid"] for alias in COMPLETED_TASK_IDS}
    ready = []
    dependencies_by_alias: dict[str, set[str]] = {alias: set() for alias in all_aliases}
    for alias, dependency in observed_dependencies:
        dependencies_by_alias[alias].add(rows_by_alias[dependency]["task_cid"])
    for alias in TODO_TASK_IDS:
        row = rows_by_alias[alias]
        if (
            row["body"].get("is_schedulable") is True
            and dependencies_by_alias[alias] <= completed_cids
        ):
            ready.append(alias)
    if ready != ["LGCVF-081"]:
        raise SuccessorOperatorError("sealed ready frontier differs")
    completion_aliases = [tasks_by_cid.get(str(row[0]), "") for row in completion_rows]
    if sorted(completion_aliases) != sorted(RECOVERED_COMPLETED_TASK_IDS):
        raise SuccessorOperatorError("sealed reconstructed completion receipts differ")
    after = _sha256_regular_file(
        database,
        noun="sealed control database",
        require_private_owner=True,
    )
    if before != after:
        raise SuccessorOperatorError(
            "sealed control database changed during verification"
        )
    identity = _database_identity(database)
    return {
        "sha256": before,
        "database_uuid": identity.get("database_uuid", ""),
        "schema_fingerprint": profile.get("schema_fingerprint", ""),
        "catalog_fingerprint": profile.get("catalog_fingerprint", ""),
        "task_count": 28,
        "dependency_count": 46,
        "completion_receipt_count": 6,
        "ready_task_ids": ready,
        "zero_state_counts": zero_counts,
        "task_cids_by_alias": {
            alias: rows_by_alias[alias]["task_cid"] for alias in sorted(rows_by_alias)
        },
    }


def _verify_sealed_coordination_state(
    database: Path,
    *,
    expected_sha256: str,
    control_tasks: Mapping[str, str],
    formal_plan: Mapping[str, Any],
) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        read_coordination_registry_projection,
    )

    before = _sha256_regular_file(
        database,
        noun="sealed coordination database",
        require_private_owner=True,
    )
    if before != expected_sha256:
        raise SuccessorOperatorError("sealed coordination database SHA-256 differs")
    try:
        projection = read_coordination_registry_projection(database)
    except Exception as exc:
        raise SuccessorOperatorError(
            f"sealed coordination projection is unreadable: {type(exc).__name__}"
        ) from exc
    expected_counts = {
        "registered_tasks": 28,
        "dependency_edges": 46,
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
    if projection.get("counts") != expected_counts or any(
        projection.get(field) != []
        for field in (
            "task_claims",
            "task_attempts",
            "fenced_leases",
            "resource_claims",
            "maintenance_leases",
        )
    ):
        raise SuccessorOperatorError("sealed coordination database has live state")
    registered = {
        str(item.get("task_id") or ""): str(item.get("task_cid") or "")
        for item in projection.get("tasks") or ()
        if isinstance(item, Mapping)
    }
    if registered != dict(control_tasks):
        raise SuccessorOperatorError("sealed coordination task registry differs")
    cid_to_alias = {cid: alias for alias, cid in control_tasks.items()}
    observed_dependencies = {
        (
            cid_to_alias.get(str(item.get("task_cid") or ""), ""),
            cid_to_alias.get(str(item.get("dependency_task_cid") or ""), ""),
        )
        for item in projection.get("dependency_edges") or ()
        if isinstance(item, Mapping)
    }
    formal_tasks = formal_plan.get("tasks") or ()
    expected_dependencies = {
        (str(task.get("task_id") or ""), str(dependency))
        for task in formal_tasks
        if isinstance(task, Mapping)
        for dependency in task.get("depends_on") or ()
    }
    completion_aliases = {
        cid_to_alias.get(str(item.get("task_cid") or ""), "")
        for item in projection.get("logical_completions") or ()
        if isinstance(item, Mapping) and item.get("status") == "succeeded"
    }
    if observed_dependencies != expected_dependencies or completion_aliases != set(
        COMPLETED_TASK_IDS
    ):
        raise SuccessorOperatorError("sealed coordination authority differs")
    after = _sha256_regular_file(
        database,
        noun="sealed coordination database",
        require_private_owner=True,
    )
    if before != after:
        raise SuccessorOperatorError(
            "sealed coordination database changed during verification"
        )
    return {"sha256": before, "counts": expected_counts}


def _verify_sealed_execution_state(
    database: Path,
    *,
    expected_sha256: str,
    control_schema_fingerprint: str,
) -> dict[str, Any]:
    import duckdb

    before = _sha256_regular_file(
        database,
        noun="sealed execution database",
        require_private_owner=True,
    )
    if before != expected_sha256:
        raise SuccessorOperatorError("sealed execution database SHA-256 differs")
    try:
        connection = duckdb.connect(
            str(database),
            read_only=True,
            config={
                "autoinstall_known_extensions": "false",
                "autoload_known_extensions": "false",
            },
        )
        try:
            counts = {
                table: int(
                    connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                )
                for table in (
                    "attempt_phases",
                    "daemon_execution_events",
                    "database_task_attempts",
                    "effect_claims",
                    "provider_invocations",
                )
            }
            metadata = {
                str(key): str(value)
                for key, value in connection.execute(
                    "SELECT key, value FROM daemon_execution_metadata ORDER BY key"
                ).fetchall()
            }
        finally:
            connection.close()
    except Exception as exc:
        raise SuccessorOperatorError(
            f"sealed execution state is unreadable: {type(exc).__name__}"
        ) from exc
    if any(counts.values()) or (
        metadata.get("authority_mode") != "embedded"
        or metadata.get("control_schema_fingerprint") != control_schema_fingerprint
        or metadata.get("control_schema_profile_id")
        != "datasets-authoritative-operational-control-plane@1"
        or metadata.get("interface") != "DatabaseImplementationDaemon@1"
        or metadata.get("process_instance_id") != "fresh-recovery-bootstrap"
        or metadata.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/database-implementation-daemon@1"
        or metadata.get("state_schema_revision")
        != "datasets-authoritative-operational-v1"
        or not str(metadata.get("logical_owner_session_id") or "").startswith(
            "embedded-store:"
        )
    ):
        raise SuccessorOperatorError("sealed execution database has unexpected state")
    after = _sha256_regular_file(
        database,
        noun="sealed execution database",
        require_private_owner=True,
    )
    if before != after:
        raise SuccessorOperatorError(
            "sealed execution database changed during verification"
        )
    return {"sha256": before, "row_counts": counts, "metadata": metadata}


def _verify_sealed_layout(paths: Mapping[str, Path], *, manifest_name: str) -> None:
    expected_root = {
        ".control.coordination.duckdb.lock",
        ".control.duckdb.intent.lock",
        ".control.duckdb.lock",
        ".control.duckdb.migration.lock",
        ".control.execution.duckdb.lock",
        "control.coordination.duckdb",
        "control.duckdb",
        "control.execution.duckdb",
        "evidence",
    }
    expected_evidence = {"bootstrap", "fresh-generation-recovery"}
    expected_bootstrap = {"materialization.json"}
    expected_recovery = {"recovery-receipt.json", manifest_name}
    observed = {
        "root": {item.name for item in os.scandir(paths["root"])},
        "evidence": {item.name for item in os.scandir(paths["root"] / "evidence")},
        "bootstrap": {
            item.name for item in os.scandir(paths["root"] / "evidence" / "bootstrap")
        },
        "recovery": {item.name for item in os.scandir(paths["recovery_root"])},
    }
    if observed != {
        "root": expected_root,
        "evidence": expected_evidence,
        "bootstrap": expected_bootstrap,
        "recovery": expected_recovery,
    }:
        raise SuccessorOperatorError("sealed run-v17 layout differs")
    for name in sorted(expected_root):
        if not name.startswith("."):
            continue
        lock_path = paths["root"] / name
        metadata = os.lstat(lock_path)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_size != 0
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise SuccessorOperatorError("sealed empty lock-file custody differs")


def _assert_sealed_report_snapshot(
    paths: Mapping[str, Path],
    report: Mapping[str, Any],
) -> None:
    pins = report.get("pins")
    manifest_cid = str(report.get("manifest_cid") or "")
    if (
        not isinstance(pins, Mapping)
        or re.fullmatch(r"bagu[a-z2-7]{20,}", manifest_cid) is None
    ):
        raise SuccessorOperatorError("sealed continuity report pins are unavailable")
    manifest = paths["recovery_root"] / f"{manifest_cid}.manifest.json"
    observed = {
        "control_sha256": _sha256_regular_file(
            paths["control"], noun="sealed control database", require_private_owner=True
        ),
        "coordination_sha256": _sha256_regular_file(
            paths["coordination"],
            noun="sealed coordination database",
            require_private_owner=True,
        ),
        "execution_sha256": _sha256_regular_file(
            paths["execution"],
            noun="sealed execution database",
            require_private_owner=True,
        ),
        "bootstrap_sha256": _sha256_regular_file(
            paths["bootstrap"],
            max_bytes=MAX_JSON_BYTES,
            noun="sealed bootstrap receipt",
            require_private_owner=True,
        ),
        "manifest_sha256": _sha256_regular_file(
            manifest,
            max_bytes=MAX_JSON_BYTES,
            noun="sealed recovery manifest",
            require_private_owner=True,
        ),
        "recovery_receipt_sha256": _sha256_regular_file(
            paths["recovery_receipt"],
            max_bytes=MAX_JSON_BYTES,
            noun="sealed recovery receipt",
            require_private_owner=True,
        ),
    }
    if any(pins.get(key) != value for key, value in observed.items()):
        raise SuccessorOperatorError(
            "sealed continuity snapshot changed after verification"
        )
    _verify_sealed_layout(paths, manifest_name=manifest.name)


def _validate_bootstrap_receipt(
    bootstrap: Mapping[str, Any],
    *,
    recovery_receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    verification = bootstrap.get("verification")
    if not isinstance(verification, Mapping):
        raise SuccessorOperatorError("sealed bootstrap verification is unavailable")
    _strict_addressed_mapping(
        verification,
        identity_field="verification_root",
        noun="sealed bootstrap verification",
    )
    expected_paths = {
        "control": (
            "data/agent_supervisor/logic_governed_compositional_verification_fabric/"
            "run-v17/control.duckdb"
        ),
        "coordination": (
            "data/agent_supervisor/logic_governed_compositional_verification_fabric/"
            "run-v17/control.coordination.duckdb"
        ),
        "execution": (
            "data/agent_supervisor/logic_governed_compositional_verification_fabric/"
            "run-v17/control.execution.duckdb"
        ),
    }
    if (
        bootstrap.get("receipt_cid") != recovery_receipt.get("bootstrap_receipt_cid")
        or bootstrap.get("population_root") != manifest.get("population_root")
        or bootstrap.get("plan_root_cid") != manifest.get("plan_root_cid")
        or bootstrap.get("source_head") != manifest.get("source_head")
        or bootstrap.get("repository_tree_id")
        != "git-tree:" + str(manifest.get("source_tree") or "")
        or bootstrap.get("authority_mode") != "embedded"
        or bootstrap.get("task_source_kind") != "duckdb"
        or bootstrap.get("maximum_writer_processes") != 1
        or bootstrap.get("quack_qualified") is not False
        or bootstrap.get("schema_revision") != "datasets-authoritative-operational-v1"
        or bootstrap.get("schema_profile") != "datasets-authoritative-operational"
        or bootstrap.get("database_paths") != expected_paths
        or verification.get("valid") is not True
        or verification.get("stores_unchanged") is not True
    ):
        raise SuccessorOperatorError("sealed bootstrap receipt binding differs")


def verify_sealed_target_continuity(
    *,
    root: Path,
    source_root: Path,
    control_sha256: str,
    coordination_sha256: str,
    execution_sha256: str,
    bootstrap_sha256: str,
    manifest_sha256: str,
    recovery_receipt_sha256: str,
) -> dict[str, Any]:
    """Admit one reviewed hash-pinned snapshot with bounded semantic checks."""

    root = root.resolve(strict=True)
    paths = _sealed_source_paths(source_root)
    pins = {
        "control_sha256": _require_sha256_pin(
            control_sha256, noun="sealed control database"
        ),
        "coordination_sha256": _require_sha256_pin(
            coordination_sha256, noun="sealed coordination database"
        ),
        "execution_sha256": _require_sha256_pin(
            execution_sha256, noun="sealed execution database"
        ),
        "bootstrap_sha256": _require_sha256_pin(
            bootstrap_sha256, noun="sealed bootstrap receipt"
        ),
        "manifest_sha256": _require_sha256_pin(
            manifest_sha256, noun="sealed recovery manifest"
        ),
        "recovery_receipt_sha256": _require_sha256_pin(
            recovery_receipt_sha256, noun="sealed recovery receipt"
        ),
    }
    if pins != SEALED_CONTINUITY_EXPECTED_PINS:
        raise SuccessorOperatorError(
            "sealed continuity pins differ from the reviewed board candidate"
        )
    recovery_receipt = _strict_addressed_json(
        paths["recovery_receipt"],
        expected_schema=FRESH_RECOVERY_RECEIPT_SCHEMA,
        identity_field="receipt_cid",
        noun="sealed recovery receipt",
    )
    manifest_cid = str(recovery_receipt.get("manifest_cid") or "")
    if re.fullmatch(r"bagu[a-z2-7]{20,}", manifest_cid) is None:
        raise SuccessorOperatorError("sealed recovery manifest CID is unsafe")
    manifest_path = paths["recovery_root"] / f"{manifest_cid}.manifest.json"
    manifest = _strict_addressed_json(
        manifest_path,
        expected_schema=FRESH_RECOVERY_MANIFEST_SCHEMA,
        identity_field="manifest_cid",
        noun="sealed recovery manifest",
    )
    bootstrap = _strict_addressed_json(
        paths["bootstrap"],
        expected_schema=BOOTSTRAP_RECEIPT_SCHEMA,
        identity_field="receipt_cid",
        noun="sealed bootstrap receipt",
    )
    observed_identities = {
        "bootstrap_receipt_cid": bootstrap.get("receipt_cid"),
        "manifest_cid": manifest.get("manifest_cid"),
        "receipt_cid": recovery_receipt.get("receipt_cid"),
        "population_root": recovery_receipt.get("population_root"),
        "source_evidence_cid": recovery_receipt.get("source_evidence_cid"),
        "sealed_operational_verification_root": recovery_receipt.get(
            "operational_verification_root"
        ),
        "target_source_head": manifest.get("source_head"),
        "target_source_tree": manifest.get("source_tree"),
    }
    if observed_identities != SEALED_CONTINUITY_EXPECTED_IDENTITIES:
        raise SuccessorOperatorError(
            "sealed continuity identities differ from the reviewed board candidate"
        )
    observed_artifact_hashes = {
        "bootstrap_sha256": _sha256_regular_file(
            paths["bootstrap"],
            max_bytes=MAX_JSON_BYTES,
            noun="sealed bootstrap receipt",
            require_private_owner=True,
        ),
        "manifest_sha256": _sha256_regular_file(
            manifest_path,
            max_bytes=MAX_JSON_BYTES,
            noun="sealed recovery manifest",
            require_private_owner=True,
        ),
        "recovery_receipt_sha256": _sha256_regular_file(
            paths["recovery_receipt"],
            max_bytes=MAX_JSON_BYTES,
            noun="sealed recovery receipt",
            require_private_owner=True,
        ),
    }
    if any(
        observed_artifact_hashes[key] != pins[key] for key in observed_artifact_hashes
    ):
        raise SuccessorOperatorError("sealed recovery artifact SHA-256 differs")
    if (
        manifest.get("manifest_cid") != manifest_cid
        or recovery_receipt.get("bootstrap_receipt_sha256") != pins["bootstrap_sha256"]
    ):
        raise SuccessorOperatorError("sealed recovery artifact cross-binding differs")
    _verify_sealed_layout(paths, manifest_name=manifest_path.name)
    _validate_bootstrap_receipt(
        bootstrap,
        recovery_receipt=recovery_receipt,
        manifest=manifest,
    )
    config = _plain_json_object(
        _contained(
            root,
            "config/agent_supervisor_logic_governed_compositional_verification_fabric_"
            "scheduler.json",
        ),
        noun="tracked scheduler config",
    )
    formal_plan = _plain_json_object(
        _contained(root, str(config.get("formal_plan_path") or "")),
        noun="tracked formal plan",
    )
    _validate_recovery_policy_projection(
        config=config,
        manifest=manifest,
        receipt=recovery_receipt,
    )
    _validate_historical_qualification(manifest)
    source_binding = _target_source_continuity(
        root,
        source_head=str(manifest.get("source_head") or ""),
        source_tree=str(manifest.get("source_tree") or ""),
        config=config,
    )
    control = _verify_sealed_control_state(
        paths["control"],
        expected_sha256=pins["control_sha256"],
        manifest=manifest,
        formal_plan=formal_plan,
    )
    coordination = _verify_sealed_coordination_state(
        paths["coordination"],
        expected_sha256=pins["coordination_sha256"],
        control_tasks=control["task_cids_by_alias"],
        formal_plan=formal_plan,
    )
    execution = _verify_sealed_execution_state(
        paths["execution"],
        expected_sha256=pins["execution_sha256"],
        control_schema_fingerprint=str(control["schema_fingerprint"]),
    )
    after_artifact_hashes = {
        "bootstrap_sha256": _sha256_regular_file(
            paths["bootstrap"],
            max_bytes=MAX_JSON_BYTES,
            noun="sealed bootstrap receipt",
            require_private_owner=True,
        ),
        "manifest_sha256": _sha256_regular_file(
            manifest_path,
            max_bytes=MAX_JSON_BYTES,
            noun="sealed recovery manifest",
            require_private_owner=True,
        ),
        "recovery_receipt_sha256": _sha256_regular_file(
            paths["recovery_receipt"],
            max_bytes=MAX_JSON_BYTES,
            noun="sealed recovery receipt",
            require_private_owner=True,
        ),
    }
    if after_artifact_hashes != observed_artifact_hashes:
        raise SuccessorOperatorError(
            "sealed recovery artifacts changed during verification"
        )
    report: dict[str, Any] = {
        "schema": SEALED_CONTINUITY_VERIFICATION_SCHEMA,
        "valid": True,
        "verification_mode": "read_only_hash_pinned_target_snapshot",
        "admission_mode": SEALED_CONTINUITY_MODE,
        "authority_ceiling": SEALED_CONTINUITY_AUTHORITY_CEILING,
        "source_root": str(paths["root"]),
        "candidate_root": str(root),
        "source_generation": "lgcvf-run-v17",
        "target_generation": "lgcvf-run-v17",
        "manifest_cid": manifest_cid,
        "receipt_cid": recovery_receipt["receipt_cid"],
        "bootstrap_receipt_cid": bootstrap["receipt_cid"],
        "source_evidence_cid": recovery_receipt["source_evidence_cid"],
        "population_root": recovery_receipt["population_root"],
        "plan_root_cid": recovery_receipt["plan_root_cid"],
        "sealed_operational_verification_root": recovery_receipt[
            "operational_verification_root"
        ],
        "pins": pins,
        "source_binding": source_binding,
        "control": {
            key: value for key, value in control.items() if key != "task_cids_by_alias"
        },
        "coordination": coordination,
        "execution": execution,
        "completed_task_ids": list(COMPLETED_TASK_IDS),
        "todo_task_ids": list(TODO_TASK_IDS),
        "blocked_task_ids": list(BLOCKED_TASK_IDS),
        "completed_count": 13,
        "todo_count": 13,
        "blocked_count": 2,
        "ready_task_ids": ["LGCVF-081"],
        "stores_unchanged": True,
        "target_database_statuses_read": True,
        "source_database_statuses_read": False,
        "fresh_source_evidence_revalidated": False,
        "historical_source_bytes_revalidated": False,
        "source_provenance_authoritative": False,
        "target_snapshot_hash_pinned": True,
        "candidate_authored_validation": True,
        "validation_self_authority": False,
        "validation_completion_authoritative": False,
        "source_database_completion_records_imported": False,
        "synthetic_source_disposition": "quarantined_not_imported",
        "network_isolation_enforced": True,
        "model_provider_route": "none",
        "task_implementation_complete": False,
        "test_qualification_complete": False,
        "objective_complete": False,
        "release_qualified": False,
        "authoritative_for_release": False,
        "production_authorized": False,
    }
    report["verification_root"] = _content_id(report)
    return report


def _canonical_recovery_verification(root: Path = ROOT) -> dict[str, Any]:
    command = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(_contained(root, MATERIALIZER_RELATIVE)),
        "recovery-verify",
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=300.0,
    )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SuccessorOperatorError(
            "canonical run-v17 recovery verifier returned malformed output"
        ) from exc
    if (
        completed.returncode != 0
        or not isinstance(report, dict)
        or report.get("valid") is not True
        or report.get("target_generation") != "lgcvf-run-v17"
        or report.get("stores_unchanged") is not True
        or report.get("source_database_statuses_read") is not False
        or report.get("completed_count") != 13
        or report.get("todo_count") != 13
        or report.get("blocked_count") != 2
        or report.get("ready_task_ids") != ["LGCVF-081"]
    ):
        raise SuccessorOperatorError(
            "canonical run-v17 recovery is not a verified 13/13/2 recovery: "
            + str(report.get("error") or completed.stderr[-1000:])
        )
    return report


def clone_verified_successor(
    source_database: Path,
    target_database: Path,
    provenance_path: Path,
    *,
    recovery_verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish one complete, verified, no-overwrite successor run."""

    source = Path(source_database).resolve(strict=True)
    target = Path(os.path.abspath(os.fspath(target_database)))
    provenance = Path(os.path.abspath(os.fspath(provenance_path)))
    final_run = target.parent
    try:
        provenance_relative = provenance.relative_to(final_run)
    except ValueError as exc:
        raise SuccessorOperatorError(
            "successor provenance must be inside the target generation"
        ) from exc
    if (
        source.parent.name != "run-v17"
        or final_run.name != "run-v18"
        or target.name != "control.duckdb"
        or len(provenance_relative.parts) != 2
        or provenance_relative.parts[0] != "evidence"
    ):
        raise SuccessorOperatorError("successor clone must be run-v17 -> run-v18")
    if source == target:
        raise SuccessorOperatorError("successor source and target are identical")
    try:
        os.lstat(final_run)
    except FileNotFoundError:
        pass
    else:
        raise SuccessorOperatorError("refusing to overwrite an existing successor")
    if os.path.lexists(source.with_name(source.name + ".wal")):
        raise SuccessorOperatorError("run-v17 control database has a live WAL")
    admission_mode = str(
        recovery_verification.get("admission_mode")
        or "canonical_fresh_generation_recovery"
    )
    sealed_source_paths: dict[str, Path] | None = None
    if admission_mode == SEALED_CONTINUITY_MODE:
        sealed_source_paths = _sealed_source_paths(
            Path(str(recovery_verification.get("source_root") or ""))
        )
        if (
            source != sealed_source_paths["control"]
            or recovery_verification.get("authority_ceiling")
            != SEALED_CONTINUITY_AUTHORITY_CEILING
            or recovery_verification.get("target_snapshot_hash_pinned") is not True
            or recovery_verification.get("historical_source_bytes_revalidated")
            is not False
            or recovery_verification.get("source_provenance_authoritative") is not False
            or recovery_verification.get("authoritative_for_release") is not False
            or recovery_verification.get("production_authorized") is not False
        ):
            raise SuccessorOperatorError(
                "sealed target continuity report is not admissible"
            )
        _require_false_authority(
            recovery_verification, noun="sealed target continuity report"
        )
        _assert_sealed_report_snapshot(sealed_source_paths, recovery_verification)
    elif admission_mode != "canonical_fresh_generation_recovery":
        raise SuccessorOperatorError("successor admission mode is unsupported")
    if (
        recovery_verification.get("valid") is not True
        or recovery_verification.get("target_generation") != "lgcvf-run-v17"
        or recovery_verification.get("stores_unchanged") is not True
        or recovery_verification.get("source_database_statuses_read") is not False
    ):
        raise SuccessorOperatorError("run-v17 recovery verification is not admissible")

    source_verification = _verify_profile(source)
    source_identity = _database_identity(source)
    source_digest = _sha256_regular_file(source)
    if sealed_source_paths is not None and source_digest != (
        recovery_verification.get("pins") or {}
    ).get("control_sha256"):
        raise SuccessorOperatorError(
            "sealed control source differs from its admitted pin"
        )

    publish_parent = final_run.parent
    publish_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _privatize_owned_directory(publish_parent, noun="successor publication parent")
    stage = publish_parent / f".{final_run.name}.{uuid.uuid4().hex}.stage"
    os.mkdir(stage, mode=0o700)
    staged_database = stage / target.name
    staged_provenance = stage / provenance_relative
    parent_descriptor = os.open(
        publish_parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    parent_before = os.fstat(parent_descriptor)
    stage_before = os.lstat(stage)
    source_descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    target_descriptor: int | None = None
    published = False
    try:
        source_before = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(source_before.st_mode)
            or source_before.st_size <= 0
            or source_before.st_size > MAX_DATABASE_BYTES
        ):
            raise SuccessorOperatorError(
                "run-v17 source is not a bounded regular database"
            )
        target_descriptor = os.open(
            staged_database,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        while True:
            block = os.read(source_descriptor, 1024 * 1024)
            if not block:
                break
            view = memoryview(block)
            while view:
                written = os.write(target_descriptor, view)
                if written <= 0:
                    raise SuccessorOperatorError("run-v18 clone write made no progress")
                view = view[written:]
        os.fsync(target_descriptor)
        os.close(target_descriptor)
        target_descriptor = None

        target_verification = _verify_profile(staged_database)
        target_identity = _database_identity(staged_database)
        target_digest = _sha256_regular_file(
            staged_database,
            noun="staged successor database",
            require_private_owner=True,
        )
        if (
            _sha256_regular_file(source) != source_digest
            or target_digest != source_digest
            or target_identity != source_identity
            or target_verification.get("schema_fingerprint")
            != source_verification.get("schema_fingerprint")
        ):
            raise SuccessorOperatorError("run-v18 clone differs from verified run-v17")
        if sealed_source_paths is not None:
            pins = recovery_verification.get("pins") or {}
            refreshed = verify_sealed_target_continuity(
                root=Path(str(recovery_verification.get("candidate_root") or "")),
                source_root=sealed_source_paths["root"],
                control_sha256=str(pins.get("control_sha256") or ""),
                coordination_sha256=str(pins.get("coordination_sha256") or ""),
                execution_sha256=str(pins.get("execution_sha256") or ""),
                bootstrap_sha256=str(pins.get("bootstrap_sha256") or ""),
                manifest_sha256=str(pins.get("manifest_sha256") or ""),
                recovery_receipt_sha256=str(pins.get("recovery_receipt_sha256") or ""),
            )
            if refreshed != dict(recovery_verification):
                raise SuccessorOperatorError(
                    "sealed continuity report changed before successor publication"
                )

        receipt = {
            "schema": PROVENANCE_SCHEMA,
            "issued_at": _utc_now(),
            "source_generation": "lgcvf-run-v17",
            "target_generation": "lgcvf-run-v18",
            "source_database": str(source),
            "target_database": str(target),
            "source_sha256": source_digest,
            "target_initial_sha256": target_digest,
            "database_uuid": source_identity.get("database_uuid", ""),
            "schema_fingerprint": source_verification["schema_fingerprint"],
            "catalog_fingerprint": source_verification["catalog_fingerprint"],
            "recovery_verification_root": str(
                recovery_verification.get("verification_root") or ""
            ),
            "recovery_receipt_cid": str(recovery_verification.get("receipt_cid") or ""),
            "recovery_manifest_cid": str(
                recovery_verification.get("manifest_cid") or ""
            ),
            "bootstrap_receipt_cid": str(
                recovery_verification.get("bootstrap_receipt_cid") or ""
            ),
            "source_evidence_cid": str(
                recovery_verification.get("source_evidence_cid") or ""
            ),
            "population_root": str(recovery_verification.get("population_root") or ""),
            "plan_root_cid": str(recovery_verification.get("plan_root_cid") or ""),
            "admission_mode": admission_mode,
            "authority_ceiling": str(
                recovery_verification.get("authority_ceiling")
                or "operational_recovery_only"
            ),
            "source_root": str(
                sealed_source_paths["root"]
                if sealed_source_paths is not None
                else source.parent
            ),
            "source_coordination_database": str(
                sealed_source_paths["coordination"]
                if sealed_source_paths is not None
                else ""
            ),
            "source_execution_database": str(
                sealed_source_paths["execution"]
                if sealed_source_paths is not None
                else ""
            ),
            "source_bootstrap_receipt": str(
                sealed_source_paths["bootstrap"]
                if sealed_source_paths is not None
                else ""
            ),
            "source_recovery_manifest": str(
                (
                    sealed_source_paths["recovery_root"]
                    / f"{recovery_verification.get('manifest_cid')}.manifest.json"
                )
                if sealed_source_paths is not None
                else ""
            ),
            "source_recovery_receipt": str(
                sealed_source_paths["recovery_receipt"]
                if sealed_source_paths is not None
                else ""
            ),
            "source_coordination_sha256": str(
                (recovery_verification.get("pins") or {}).get("coordination_sha256")
                if sealed_source_paths is not None
                else ""
            ),
            "source_execution_sha256": str(
                (recovery_verification.get("pins") or {}).get("execution_sha256")
                if sealed_source_paths is not None
                else ""
            ),
            "source_bootstrap_sha256": str(
                (recovery_verification.get("pins") or {}).get("bootstrap_sha256")
                if sealed_source_paths is not None
                else ""
            ),
            "source_recovery_manifest_sha256": str(
                (recovery_verification.get("pins") or {}).get("manifest_sha256")
                if sealed_source_paths is not None
                else ""
            ),
            "source_recovery_receipt_sha256": str(
                (recovery_verification.get("pins") or {}).get("recovery_receipt_sha256")
                if sealed_source_paths is not None
                else ""
            ),
            "target_source_head": str(
                (recovery_verification.get("source_binding") or {}).get(
                    "target_source_head"
                )
                if sealed_source_paths is not None
                else ""
            ),
            "target_source_tree": str(
                (recovery_verification.get("source_binding") or {}).get(
                    "target_source_tree"
                )
                if sealed_source_paths is not None
                else ""
            ),
            "sealed_operational_verification_root": str(
                recovery_verification.get("sealed_operational_verification_root") or ""
            ),
            "fresh_source_evidence_revalidated": admission_mode
            != SEALED_CONTINUITY_MODE,
            "historical_source_bytes_revalidated": admission_mode
            != SEALED_CONTINUITY_MODE,
            "source_provenance_authoritative": admission_mode != SEALED_CONTINUITY_MODE,
            "target_snapshot_hash_pinned": admission_mode == SEALED_CONTINUITY_MODE,
            "target_database_statuses_read": admission_mode == SEALED_CONTINUITY_MODE,
            "source_database_statuses_read_scope": (
                "lost_fresh_recovery_source_generation_lgcvf-run-v16"
            ),
            "restart_requires_live_continuity_receipt": admission_mode
            == SEALED_CONTINUITY_MODE,
            "live_continuity_receipt_implemented": False,
            "clone_preserves_database_uuid": True,
            "owner_generation_rotates_on_start": True,
            "source_database_statuses_read": False,
            "source_database_completion_records_imported": False,
            "candidate_authored_validation": True,
            "validation_self_authority": False,
            "validation_completion_authoritative": False,
            "synthetic_source_disposition": "quarantined_not_imported",
            "network_isolation_enforced": True,
            "model_provider_route": "none",
            "task_implementation_complete": False,
            "test_qualification_complete": False,
            "objective_complete": False,
            "release_qualified": False,
            "authoritative_for_release": False,
            "production_authorized": False,
        }
        receipt["receipt_cid"] = _content_id(receipt)
        _atomic_json(staged_provenance, receipt, replace=False)
        if (
            _strict_json(
                staged_provenance,
                expected_schema=PROVENANCE_SCHEMA,
                require_private_owner=True,
            )
            != receipt
        ):
            raise SuccessorOperatorError("staged successor provenance differs")
        _remove_staged_database_locks(stage, staged_database.name)
        if (
            {item.name for item in os.scandir(stage)}
            != {target.name, provenance_relative.parts[0]}
            or {item.name for item in os.scandir(staged_provenance.parent)}
            != {staged_provenance.name}
            or os.path.lexists(staged_database.with_name(staged_database.name + ".wal"))
        ):
            raise SuccessorOperatorError("staged successor inventory differs")
        _require_private_directory(stage, noun="staged successor generation")
        _require_private_directory(
            staged_provenance.parent, noun="staged successor evidence"
        )
        source_after = os.fstat(source_descriptor)
        stage_after = os.lstat(stage)
        parent_after = os.fstat(parent_descriptor)
        if (
            (
                source_before.st_dev,
                source_before.st_ino,
                source_before.st_size,
                source_before.st_mtime_ns,
                source_before.st_ctime_ns,
            )
            != (
                source_after.st_dev,
                source_after.st_ino,
                source_after.st_size,
                source_after.st_mtime_ns,
                source_after.st_ctime_ns,
            )
            or (stage_before.st_dev, stage_before.st_ino)
            != (stage_after.st_dev, stage_after.st_ino)
            or (parent_before.st_dev, parent_before.st_ino)
            != (parent_after.st_dev, parent_after.st_ino)
            or _sha256_regular_file(
                staged_database,
                noun="staged successor database",
                require_private_owner=True,
            )
            != target_digest
            or _sha256_regular_file(source) != source_digest
        ):
            raise SuccessorOperatorError(
                "source or staged successor changed before publication"
            )
        stage_descriptor = os.open(
            stage,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(stage_descriptor)
        finally:
            os.close(stage_descriptor)
        _rename_directory_noreplace(parent_descriptor, stage.name, final_run.name)
        published = True
        try:
            os.fsync(parent_descriptor)
        except OSError as exc:
            raise SuccessorOperatorError(
                "successor published completely but parent durability is uncertain"
            ) from exc
        return receipt
    finally:
        if target_descriptor is not None:
            os.close(target_descriptor)
        os.close(source_descriptor)
        os.close(parent_descriptor)
        if not published:
            _cleanup_successor_stage(
                stage,
                staged_database=staged_database,
                staged_provenance=staged_provenance,
            )


def _require_ignored_successor(root: Path) -> None:
    _git_quiet(
        root,
        (
            "check-ignore",
            "-q",
            "--no-index",
            str(SUCCESSOR_DATABASE_RELATIVE),
        ),
        noun="run-v18 successor Git-ignore policy",
    )


def bootstrap_successor(root: Path = ROOT) -> dict[str, Any]:
    paths = _paths(root)
    _require_ignored_successor(root)
    recovery = _canonical_recovery_verification(root)
    return clone_verified_successor(
        paths["source_database"],
        paths["successor_database"],
        paths["provenance"],
        recovery_verification=recovery,
    )


def bootstrap_sealed_successor(
    *,
    root: Path,
    source_root: Path,
    control_sha256: str,
    coordination_sha256: str,
    execution_sha256: str,
    bootstrap_sha256: str,
    manifest_sha256: str,
    recovery_receipt_sha256: str,
) -> dict[str, Any]:
    paths = _paths(root)
    _require_ignored_successor(root)
    verification = verify_sealed_target_continuity(
        root=root,
        source_root=source_root,
        control_sha256=control_sha256,
        coordination_sha256=coordination_sha256,
        execution_sha256=execution_sha256,
        bootstrap_sha256=bootstrap_sha256,
        manifest_sha256=manifest_sha256,
        recovery_receipt_sha256=recovery_receipt_sha256,
    )
    source_paths = _sealed_source_paths(source_root)
    return clone_verified_successor(
        source_paths["control"],
        paths["successor_database"],
        paths["provenance"],
        recovery_verification=verification,
    )


def _load_provenance(paths: Mapping[str, Path], *, root: Path = ROOT) -> dict[str, Any]:
    database = paths["successor_database"]
    _require_private_directory(database.parent, noun="successor generation")
    _require_private_directory(
        paths["provenance"].parent, noun="successor evidence directory"
    )
    if os.path.lexists(database.with_name(database.name + ".wal")):
        raise SuccessorOperatorError("successor control database has a live WAL")
    receipt = _strict_json(
        paths["provenance"],
        expected_schema=PROVENANCE_SCHEMA,
        require_private_owner=True,
    )
    target_digest = _sha256_regular_file(
        database,
        noun="successor control database",
        require_private_owner=True,
    )
    admission_mode = str(receipt.get("admission_mode") or "")
    if receipt.get("target_database") != str(database):
        raise SuccessorOperatorError("successor provenance target differs")
    if admission_mode == "canonical_fresh_generation_recovery":
        source_database = paths["source_database"]
        if receipt.get("source_database") != str(source_database):
            raise SuccessorOperatorError("successor provenance no longer binds run-v17")
    elif admission_mode == SEALED_CONTINUITY_MODE:
        if (
            receipt.get("authority_ceiling") != SEALED_CONTINUITY_AUTHORITY_CEILING
            or receipt.get("fresh_source_evidence_revalidated") is not False
            or receipt.get("historical_source_bytes_revalidated") is not False
            or receipt.get("source_provenance_authoritative") is not False
            or receipt.get("target_snapshot_hash_pinned") is not True
            or receipt.get("target_database_statuses_read") is not True
            or receipt.get("source_database_statuses_read_scope")
            != "lost_fresh_recovery_source_generation_lgcvf-run-v16"
            or receipt.get("restart_requires_live_continuity_receipt") is not True
            or receipt.get("live_continuity_receipt_implemented") is not False
            or receipt.get("authoritative_for_release") is not False
            or receipt.get("production_authorized") is not False
            or receipt.get("source_generation") != "lgcvf-run-v17"
            or receipt.get("target_generation") != "lgcvf-run-v18"
            or receipt.get("clone_preserves_database_uuid") is not True
            or receipt.get("owner_generation_rotates_on_start") is not True
        ):
            raise SuccessorOperatorError("sealed successor authority ceiling differs")
        _require_false_authority(receipt, noun="sealed successor provenance")
        sealed = _sealed_source_paths(Path(str(receipt.get("source_root") or "")))
        source_database = sealed["control"]
        expected_manifest = (
            sealed["recovery_root"]
            / f"{receipt.get('recovery_manifest_cid')}.manifest.json"
        )
        expected_paths = {
            "source_database": sealed["control"],
            "source_coordination_database": sealed["coordination"],
            "source_execution_database": sealed["execution"],
            "source_bootstrap_receipt": sealed["bootstrap"],
            "source_recovery_receipt": sealed["recovery_receipt"],
            "source_recovery_manifest": expected_manifest,
        }
        if any(
            receipt.get(field) != str(path) for field, path in expected_paths.items()
        ):
            raise SuccessorOperatorError("sealed successor source path binding differs")
        sealed_hashes = {
            "source_sha256": _sha256_regular_file(
                sealed["control"],
                noun="sealed control database",
                require_private_owner=True,
            ),
            "source_coordination_sha256": _sha256_regular_file(
                sealed["coordination"],
                noun="sealed coordination database",
                require_private_owner=True,
            ),
            "source_execution_sha256": _sha256_regular_file(
                sealed["execution"],
                noun="sealed execution database",
                require_private_owner=True,
            ),
            "source_bootstrap_sha256": _sha256_regular_file(
                sealed["bootstrap"],
                max_bytes=MAX_JSON_BYTES,
                noun="sealed bootstrap receipt",
                require_private_owner=True,
            ),
            "source_recovery_manifest_sha256": _sha256_regular_file(
                expected_manifest,
                max_bytes=MAX_JSON_BYTES,
                noun="sealed recovery manifest",
                require_private_owner=True,
            ),
            "source_recovery_receipt_sha256": _sha256_regular_file(
                sealed["recovery_receipt"],
                max_bytes=MAX_JSON_BYTES,
                noun="sealed recovery receipt",
                require_private_owner=True,
            ),
        }
        if any(receipt.get(field) != digest for field, digest in sealed_hashes.items()):
            raise SuccessorOperatorError("sealed successor source hash binding differs")
        if sealed_hashes != {
            "source_sha256": SEALED_CONTINUITY_EXPECTED_PINS["control_sha256"],
            "source_coordination_sha256": SEALED_CONTINUITY_EXPECTED_PINS[
                "coordination_sha256"
            ],
            "source_execution_sha256": SEALED_CONTINUITY_EXPECTED_PINS[
                "execution_sha256"
            ],
            "source_bootstrap_sha256": SEALED_CONTINUITY_EXPECTED_PINS[
                "bootstrap_sha256"
            ],
            "source_recovery_manifest_sha256": SEALED_CONTINUITY_EXPECTED_PINS[
                "manifest_sha256"
            ],
            "source_recovery_receipt_sha256": SEALED_CONTINUITY_EXPECTED_PINS[
                "recovery_receipt_sha256"
            ],
        }:
            raise SuccessorOperatorError("sealed successor reviewed pins differ")
        refreshed = verify_sealed_target_continuity(
            root=root,
            source_root=sealed["root"],
            **SEALED_CONTINUITY_EXPECTED_PINS,
        )
        source_binding = refreshed.get("source_binding") or {}
        semantic_bindings = {
            "recovery_verification_root": refreshed.get("verification_root"),
            "recovery_receipt_cid": refreshed.get("receipt_cid"),
            "recovery_manifest_cid": refreshed.get("manifest_cid"),
            "bootstrap_receipt_cid": refreshed.get("bootstrap_receipt_cid"),
            "source_evidence_cid": refreshed.get("source_evidence_cid"),
            "population_root": refreshed.get("population_root"),
            "plan_root_cid": refreshed.get("plan_root_cid"),
            "target_source_head": source_binding.get("target_source_head"),
            "target_source_tree": source_binding.get("target_source_tree"),
            "sealed_operational_verification_root": refreshed.get(
                "sealed_operational_verification_root"
            ),
        }
        if any(
            receipt.get(field) != expected
            for field, expected in semantic_bindings.items()
        ):
            raise SuccessorOperatorError(
                "sealed successor provenance cross-binding differs"
            )
        if (
            target_digest != receipt.get("target_initial_sha256")
            or target_digest != SEALED_CONTINUITY_EXPECTED_PINS["control_sha256"]
        ):
            raise SuccessorOperatorError(
                "sealed successor changed after its initial admission; restart "
                "requires an unimplemented live-continuity receipt"
            )
    else:
        raise SuccessorOperatorError("successor provenance admission mode differs")
    if _sha256_regular_file(
        source_database,
        noun="successor provenance source database",
        require_private_owner=admission_mode == SEALED_CONTINUITY_MODE,
    ) != receipt.get("source_sha256"):
        raise SuccessorOperatorError("successor provenance no longer binds run-v17")
    verification = _verify_profile(database)
    identity = _database_identity(database)
    if (
        verification.get("schema_fingerprint") != receipt.get("schema_fingerprint")
        or verification.get("catalog_fingerprint") != receipt.get("catalog_fingerprint")
        or identity.get("database_uuid") != receipt.get("database_uuid")
    ):
        raise SuccessorOperatorError(
            "successor database identity differs from provenance"
        )
    return receipt


def _parse_quack_endpoint(endpoint: str) -> tuple[str, int]:
    match = re.fullmatch(r"quack:(?://)?(127\.0\.0\.1|localhost):(\d{1,5})", endpoint)
    if match is None or not 1 <= int(match.group(2)) <= 65535:
        raise SuccessorOperatorError("successor Quack endpoint must be fixed loopback")
    return match.group(1), int(match.group(2))


def _validate_successor_board(
    config_path: Path, root: Path = ROOT
) -> tuple[Any, Any, str, int]:
    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        load_configured_board,
        preflight_configured_board,
    )

    board = load_configured_board(config_path, repo_root=root)
    program = board.resolved_database_program()
    raw_program = board.payload.get("database_program")
    expected_store = SUCCESSOR_DATABASE_RELATIVE.as_posix()
    expected_registry = OWNER_STATE_RELATIVE.as_posix()
    provider = board.payload.get("provider")
    bootstrap = board.payload.get("bootstrap_writer_policy")
    projection = board.payload.get("ducklake_projection_program")
    if (
        board.max_lanes != 4
        or board.board_namespace
        != "logic-governed-compositional-verification-fabric-v1"
        or program.authority_mode != "quack"
        or program.task_source_kind != "duckdb"
        or program.failover_policy != "fail_closed"
        or program.endpoint_secret_handle != SECRET_HANDLE
        or program.store_id != expected_store
        or program.runtime_registry_path != expected_registry
        or program.store_generation != "lgcvf-run-v18"
        or program.schema_revision != "datasets-authoritative-operational-v1"
        or not isinstance(raw_program, Mapping)
        or raw_program.get("schema_profile") != "datasets-authoritative-operational"
        or board.runtime_paths.get("root") != SUCCESSOR_RUN_RELATIVE.as_posix()
        or not isinstance(provider, Mapping)
        or provider.get("primary_provider_id") != "grok_cli"
        or provider.get("primary_model_id") != "grok-4.6"
        or provider.get("fallback_provider_id") != "codex"
        or provider.get("fallback_model_id") != "gpt-5.6-terra"
        or provider.get("fallback_trigger") != "primary_quota_exhausted"
        or provider.get("fallback_reasoning_effort") != "high"
        or provider.get("max_concurrency") != 4
        or not isinstance(bootstrap, Mapping)
        or bootstrap.get("maximum_processes") != 1
        or bootstrap.get("quack_required") is not True
        or bootstrap.get("direct_multi_process_duckdb_permitted") is not False
        or not isinstance(projection, Mapping)
        or projection.get("root") != PROJECTION_ROOT_RELATIVE.as_posix()
        or projection.get("catalog_path")
        != (PROJECTION_ROOT_RELATIVE / "lake.ducklake").as_posix()
        or projection.get("data_path")
        != (PROJECTION_ROOT_RELATIVE / "lake-data").as_posix()
        or projection.get("authority") is not False
        or projection.get("scheduling_prerequisite") is not False
        or projection.get("completion_prerequisite") is not False
        or "fresh_generation_recovery" in board.payload
    ):
        raise SuccessorOperatorError(
            "scheduler config is not the exact four-lane successor"
        )
    host, port = _parse_quack_endpoint(program.quack_endpoint)
    preflight = preflight_configured_board(board)
    if preflight.get("valid") is not True:
        raise SuccessorOperatorError(
            "configured-board preflight failed: "
            + ", ".join(preflight.get("errors") or ())
        )
    return board, program, host, port


def _status_payload(
    *,
    lifecycle: str,
    controller_birth: Mapping[str, Any],
    provenance_cid: str,
    owner_identity: Mapping[str, Any] | None = None,
    scheduler_birth: Mapping[str, Any] | None = None,
    scheduler_returncode: int | None = None,
    error: str = "",
    projection_root: Path | None = None,
) -> dict[str, Any]:
    observed_projection_root = (
        _paths()["projection_root"]
        if projection_root is None
        else Path(projection_root).resolve()
    )
    payload: dict[str, Any] = {
        "schema": CONTROLLER_STATUS_SCHEMA,
        "lifecycle": lifecycle,
        "updated_at": _utc_now(),
        "controller_birth": dict(controller_birth),
        "provenance_cid": provenance_cid,
        "owner_identity": dict(owner_identity or {}),
        "scheduler_birth": dict(scheduler_birth or {}),
        "scheduler_returncode": scheduler_returncode,
        "error": error,
        "ducklake_projection": {
            "path": str(observed_projection_root),
            "control_catalog_path": str(observed_projection_root / "control.duckdb"),
            "ducklake_catalog_path": str(observed_projection_root / "lake.ducklake"),
            "ducklake_data_path": str(observed_projection_root / "lake-data"),
            "authoritative": False,
            "read_by_scheduler": False,
            "scheduling_authority": False,
            "completion_authority": False,
            "live_quack_endpoint": False,
            "mode": "separate_stopped_checkpoint",
        },
    }
    payload["status_cid"] = _content_id(payload)
    return payload


def _write_status(path: Path, payload: Mapping[str, Any], *, token: str = "") -> None:
    encoded = _canonical_bytes(payload)
    if token and token.encode("ascii") in encoded:
        raise SuccessorOperatorError("Quack token would enter controller status")
    _atomic_json(path, payload, replace=True)


def _token_sink(owner_state: Path) -> Path:
    """Return an impossible child path so legacy helpers cannot persist the token."""

    marker = owner_state / ".ephemeral-token-persistence-disabled"
    payload = b"trusted controller keeps the Quack attach credential in memory\n"
    if marker.exists():
        observed = os.lstat(marker)
        if not stat.S_ISREG(observed.st_mode) or marker.read_bytes() != payload:
            raise SuccessorOperatorError("ephemeral token sink marker is unsafe")
    else:
        descriptor = os.open(
            marker,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(marker, 0o400)
    # The parent component is a regular file, so mkdir/open in the legacy
    # persistence helper fails without ever creating credential material.
    return marker / "unavailable"


def _prepare_private_owner_socket(socket_path: Path) -> None:
    """Admit one short same-UID directory without following a symlink."""

    path = Path(socket_path)
    temporary_root = Path(tempfile.gettempdir()).resolve()
    parent = path.parent
    if (
        not path.is_absolute()
        or parent.parent.resolve() != temporary_root
        or parent.name != f"ipfs-accelerate-lgcvf-{os.geteuid()}"
        or not path.name.startswith("owner-")
        or not path.name.endswith(".sock")
        or len(os.fsencode(path)) > UNIX_SOCKET_PATH_CEILING
    ):
        raise SuccessorOperatorError("state-owner socket identity is unsafe")
    try:
        parent.mkdir(mode=0o700)
    except FileExistsError:
        pass
    try:
        metadata = os.lstat(parent)
    except OSError as exc:
        raise SuccessorOperatorError(
            "state-owner socket directory is unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise SuccessorOperatorError("state-owner socket directory custody is unsafe")
    try:
        existing = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SuccessorOperatorError("state-owner socket cannot be inspected") from exc
    if (
        not stat.S_ISSOCK(existing.st_mode)
        or stat.S_ISLNK(existing.st_mode)
        or existing.st_uid != os.geteuid()
        or existing.st_nlink != 1
        or stat.S_IMODE(existing.st_mode) & 0o077
    ):
        raise SuccessorOperatorError("existing state-owner socket custody is unsafe")


def _child_environment(
    *,
    token: str,
    identity: Any,
    owner_state: Path,
    root: Path,
    rendered_environment: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ)
    rendered = dict(rendered_environment or {})
    if TOKEN_ENV in rendered or TOKEN_FILE_ENV in rendered:
        raise SuccessorOperatorError(
            "configured scheduler rendered an attach-credential environment field"
        )
    environment.update({str(name): str(value) for name, value in rendered.items()})
    environment[TOKEN_ENV] = token
    environment[TOKEN_FILE_ENV] = str(_token_sink(owner_state))
    environment["IPFS_ACCELERATE_AGENT_STATE_STORE_LIVE_GENERATION"] = str(
        identity.generation
    )
    environment["IPFS_ACCELERATE_AGENT_STATE_LIVE_SCHEMA_REVISION"] = str(
        identity.schema_revision
    )
    environment["IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT"] = str(root)
    environment[BOARD_EXTENSION_INSTALL_POLICY_ENV] = (
        BOARD_EXTENSION_INSTALL_POLICY_LOAD_ONLY
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _exact_birth(pid: int) -> Any:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        read_process_birth,
    )

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        birth = read_process_birth(pid)
        if birth is not None:
            return birth
        time.sleep(0.01)
    raise SuccessorOperatorError("could not capture scheduler process birth")


def _terminate_exact(
    birth: Any,
    *,
    grace_seconds: float = 10.0,
    child_process: subprocess.Popen[Any] | None = None,
) -> str:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        OwnerLiveness,
        owner_liveness,
    )

    def send(signum: int) -> None:
        if child_process is not None and child_process.poll() is not None:
            return
        state = owner_liveness(birth)
        if state is OwnerLiveness.DEAD:
            return
        if state is not OwnerLiveness.ALIVE:
            raise SuccessorOperatorError("scheduler birth is uninspectable")
        if birth.pid <= 1:
            raise SuccessorOperatorError("refusing to signal an unsafe PID")
        try:
            group = os.getpgid(birth.pid)
            if group == birth.pid:
                os.killpg(group, signum)
            else:
                os.kill(birth.pid, signum)
        except ProcessLookupError:
            return

    if child_process is not None:
        if child_process.pid != birth.pid:
            raise SuccessorOperatorError("scheduler child differs from its birth")
        if child_process.poll() is not None:
            return "already_dead"
    if owner_liveness(birth) is OwnerLiveness.DEAD:
        return "already_dead"
    send(signal.SIGTERM)
    deadline = time.monotonic() + max(0.1, grace_seconds)
    while time.monotonic() < deadline:
        if child_process is not None and child_process.poll() is not None:
            return "terminated"
        state = owner_liveness(birth)
        if state is OwnerLiveness.DEAD:
            return "terminated"
        if state is OwnerLiveness.UNKNOWN:
            raise SuccessorOperatorError("scheduler became uninspectable during stop")
        time.sleep(0.05)
    send(signal.SIGKILL)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if child_process is not None and child_process.poll() is not None:
            return "killed"
        state = owner_liveness(birth)
        if state is OwnerLiveness.DEAD:
            return "killed"
        if state is OwnerLiveness.UNKNOWN:
            raise SuccessorOperatorError("scheduler became uninspectable after kill")
        time.sleep(0.05)
    raise SuccessorOperatorError("exact scheduler birth survived bounded stop")


def run_successor(
    config_path: Path,
    *,
    root: Path = ROOT,
    implement: bool,
    duration_seconds: float,
) -> int:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        current_process_birth,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        configured_board_launch_plan,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.process_security import (
        harden_state_authority_process,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        build_server,
    )

    paths = _paths(root)
    provenance = _load_provenance(paths, root=root)
    board, program, host, port = _validate_successor_board(config_path, root)
    rendered_plan = configured_board_launch_plan(
        board,
        implement=implement,
        detach=False,
        duration_seconds=duration_seconds,
    )
    rendered_environment = rendered_plan.get("environment")
    if not isinstance(rendered_environment, Mapping):
        raise SuccessorOperatorError("configured scheduler environment is unavailable")
    expected_route_environment = {
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER": "grok_cli",
        "IPFS_ACCELERATE_AGENT_GROK_MODEL": "grok-4.6",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER": "codex",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER": (
            "primary_quota_exhausted"
        ),
        "IPFS_ACCELERATE_AGENT_CODEX_MODEL": "gpt-5.6-terra",
        "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT": "high",
    }
    if any(
        rendered_environment.get(name) != value
        for name, value in expected_route_environment.items()
    ):
        raise SuccessorOperatorError(
            "configured scheduler did not render the reviewed ordered provider route"
        )
    # The owner-command dispatcher validates the board's logical generation
    # independently of the live integer server generation.
    os.environ[STORE_GENERATION_ENV] = program.store_generation
    paths["owner_state"].mkdir(mode=0o700, parents=True, exist_ok=True)
    _prepare_private_owner_socket(paths["owner_socket"])
    lock_handle = _open_private_lock(paths["controller_lock"])
    server: Any | None = None

    def stop_owner() -> Mapping[str, Any]:
        nonlocal server
        if server is None:
            return {"stopped": True, "already_stopped": True}
        owned_server = server
        server = None
        return owned_server.stop()

    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SuccessorOperatorError(
                "another successor controller owns the lock"
            ) from exc

        controller_birth = current_process_birth()
        server = build_server(
            database_path=paths["successor_database"],
            state_dir=paths["owner_state"],
            host=host,
            port=port,
            repository_id="repository:lgcvf-quack-successor",
            store_id=program.store_id,
            secret_handle=program.endpoint_secret_handle,
            migrate=datasets_profile_migration,
            typed_command_socket_path=paths["owner_socket"],
        )
        if server.typed_command_socket_path() != paths["owner_socket"]:
            raise SuccessorOperatorError("owner did not retain its short socket path")
        identity = server.start()
        if identity.listen_uri != program.quack_endpoint:
            stop_owner()
            raise SuccessorOperatorError(
                "owner endpoint differs from scheduler program"
            )
        if server._vault is None:
            stop_owner()
            raise SuccessorOperatorError("owner token vault is unavailable")
        token = server._vault.resolve(identity.secret_handle)
        # Harden without copying the credential into the controller environment.
        harden_state_authority_process({TOKEN_ENV: token})
        token_path = paths["owner_state"] / (
            identity.secret_handle.replace(":", "_").replace("/", "_") + ".quack-token"
        )
        if token_path.exists() or token.encode("ascii") in _canonical_bytes(
            server.status()
        ):
            stop_owner()
            raise SuccessorOperatorError("owner published its Quack attach token")

        command = [
            sys.executable,
            str(
                _contained(
                    root, "scripts/ops/agent_supervisor/configured_board_scheduler.py"
                )
            ),
            "--repo-root",
            str(root),
            "--config",
            str(config_path),
            "launch",
            "--foreground",
            "--duration-seconds",
            str(duration_seconds),
        ]
        if implement:
            command.append("--implement")
        if any(token in item for item in command):
            stop_owner()
            raise SuccessorOperatorError("scheduler argv would contain the Quack token")
        environment = _child_environment(
            token=token,
            identity=identity,
            owner_state=paths["owner_state"],
            root=root,
            rendered_environment=rendered_environment,
        )
        paths["controller_log"].parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        log_handle = paths["controller_log"].open("ab")
        os.chmod(paths["controller_log"], 0o600)
        scheduler: subprocess.Popen[Any] | None = None
        scheduler_birth: Any | None = None
        stop_requested = False
        prior_handlers: dict[int, Any] = {}

        def request_stop(_signum: int, _frame: Any) -> None:
            nonlocal stop_requested
            stop_requested = True

        try:
            scheduler = subprocess.Popen(
                command,
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            scheduler_birth = _exact_birth(scheduler.pid)
            for signum in (signal.SIGINT, signal.SIGTERM):
                prior_handlers[signum] = signal.signal(signum, request_stop)
            ready_status = _status_payload(
                lifecycle="ready",
                controller_birth=controller_birth.to_dict(),
                provenance_cid=str(provenance["receipt_cid"]),
                owner_identity=identity.to_dict(),
                scheduler_birth=scheduler_birth.to_dict(),
                projection_root=paths["projection_root"],
            )
            _write_status(paths["controller_status"], ready_status, token=token)
            started = time.monotonic()
            pump_error = ""
            while scheduler.poll() is None and not stop_requested:
                if duration_seconds != float("inf") and (
                    time.monotonic() - started >= duration_seconds
                ):
                    stop_requested = True
                    break
                try:
                    server.service_mutation_inbox(max_requests=32)
                except Exception as exc:  # noqa: BLE001 - owner pump fails closed.
                    pump_error = f"{type(exc).__name__}: {exc}"
                    stop_requested = True
                    break
                time.sleep(0.01)
            if stop_requested and scheduler.poll() is None:
                _terminate_exact(
                    scheduler_birth,
                    grace_seconds=10.0,
                    child_process=scheduler,
                )
            returncode = scheduler.wait(timeout=5.0)
            if pump_error:
                raise SuccessorOperatorError(
                    "mutation inbox pump failed: " + pump_error
                )
        finally:
            for signum, handler in prior_handlers.items():
                signal.signal(signum, handler)
            if (
                scheduler is not None
                and scheduler.poll() is None
                and scheduler_birth is not None
            ):
                _terminate_exact(
                    scheduler_birth,
                    grace_seconds=5.0,
                    child_process=scheduler,
                )
                scheduler.wait(timeout=5.0)
            log_handle.close()
            stop_receipt = stop_owner()
            credential_leak = bool(tuple(paths["owner_state"].glob("*.quack-token")))
            for surface in (
                paths["controller_log"],
                paths["controller_status"],
                paths["owner_state"] / "quack-state-server.status.json",
            ):
                credential_leak = credential_leak or _regular_file_contains(
                    surface,
                    token.encode("ascii"),
                )
            stopped = _status_payload(
                lifecycle="stopped",
                controller_birth=controller_birth.to_dict(),
                provenance_cid=str(provenance["receipt_cid"]),
                owner_identity=identity.to_dict(),
                scheduler_birth=(
                    scheduler_birth.to_dict() if scheduler_birth is not None else {}
                ),
                scheduler_returncode=(
                    scheduler.returncode if scheduler is not None else None
                ),
                error=(
                    "attach_credential_persisted"
                    if credential_leak
                    else "" if stop_receipt.get("stopped") else "owner_stop_failed"
                ),
                projection_root=paths["projection_root"],
            )
            _write_status(paths["controller_status"], stopped, token=token)
            token = ""
            if credential_leak:
                raise SuccessorOperatorError(
                    "raw Quack attach credential reached a persistent surface"
                )
        return int(returncode)
    finally:
        if server is not None:
            try:
                stop_owner()
            except Exception as cleanup_exc:  # noqa: BLE001
                sys.stderr.write(
                    "LGCVF owner emergency stop failed: "
                    f"{type(cleanup_exc).__name__}\n"
                )
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()


def controller_status(root: Path = ROOT) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        OwnerLiveness,
        ProcessBirthIdentity,
        owner_liveness,
    )

    paths = _paths(root)
    status = _strict_json(
        paths["controller_status"],
        expected_schema=CONTROLLER_STATUS_SCHEMA,
        require_private_owner=True,
    )
    birth = ProcessBirthIdentity.from_dict(status.get("controller_birth"))
    observed = owner_liveness(birth)
    projection = dict(status.get("ducklake_projection") or {})
    projection["receipt_present"] = paths["projection_receipt"].is_file()
    return {
        **status,
        "observed_controller_liveness": observed.value,
        "running": observed is OwnerLiveness.ALIVE
        and status.get("lifecycle") == "ready",
        "ducklake_projection": projection,
    }


def stop_controller(
    root: Path = ROOT, *, timeout_seconds: float = MAX_STOP_SECONDS
) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        ProcessBirthIdentity,
    )

    status = controller_status(root)
    birth = ProcessBirthIdentity.from_dict(status.get("controller_birth"))
    disposition = _terminate_exact(birth, grace_seconds=min(timeout_seconds, 15.0))
    return {
        "stopped": True,
        "disposition": disposition,
        "controller_birth": birth.to_dict(),
    }


def _extension_preflight() -> dict[str, Any]:
    try:
        import duckdb

        connection = duckdb.connect(":memory:")
        try:
            connection.execute("SET autoinstall_known_extensions = false")
            connection.execute("SET autoload_known_extensions = false")
            loaded: dict[str, str] = {}
            for extension in ("quack", "ducklake"):
                connection.execute(f"LOAD {extension}")
                row = connection.execute(
                    "SELECT installed, loaded, extension_version FROM duckdb_extensions() "
                    "WHERE extension_name = ?",
                    [extension],
                ).fetchone()
                if row is None or row[0] is not True or row[1] is not True:
                    raise SuccessorOperatorError(f"{extension} is not preinstalled")
                loaded[extension] = str(row[2] or "")
        finally:
            connection.close()
    except Exception as exc:  # noqa: BLE001 - capability is typed unavailable.
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "automatic_installation_permitted": False,
        }
    return {
        "available": True,
        "extensions": loaded,
        "automatic_installation_permitted": False,
    }


def _controller_lock_is_held(path: Path) -> bool:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise SuccessorOperatorError("controller lock cannot be inspected") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise SuccessorOperatorError("controller lock custody is unsafe")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def projection_preflight(
    root: Path = ROOT,
    *,
    _checkpoint_lock_held: bool = False,
) -> dict[str, Any]:
    paths = _paths(root)
    lock_held = (
        True
        if _checkpoint_lock_held
        else _controller_lock_is_held(paths["controller_lock"])
    )
    running = lock_held and not _checkpoint_lock_held
    try:
        running = running or bool(controller_status(root).get("running"))
    except SuccessorOperatorError:
        pass
    capability = _extension_preflight()
    source_admitted = False
    source_error = ""
    if not running:
        try:
            _load_provenance(paths, root=root)
            source_admitted = True
        except (OSError, RuntimeError, ValueError) as exc:
            source_error = f"{type(exc).__name__}: {exc}"
    return {
        "schema": PROJECTION_RECEIPT_SCHEMA,
        "valid": (
            capability.get("available") is True and not running and source_admitted
        ),
        "projection_root": str(paths["projection_root"]),
        "control_catalog_path": str(paths["projection_root"] / "control.duckdb"),
        "ducklake_catalog_path": str(paths["projection_root"] / "lake.ducklake"),
        "ducklake_data_path": str(paths["projection_root"] / "lake-data"),
        "source_database": str(paths["successor_database"]),
        "controller_running": running,
        "controller_lock_held": lock_held,
        "source_database_present": paths["successor_database"].is_file(),
        "provenance_receipt_present": paths["provenance"].is_file(),
        "source_admitted": source_admitted,
        "source_error": source_error,
        "requires_stopped_checkpoint": True,
        "capability": capability,
        "authoritative": False,
        "scheduling_authority": False,
        "completion_authority": False,
        "read_by_scheduler": False,
        "quack_endpoint_served": False,
        "separate_projection_reason": (
            "BoardControlPlane owns a distinct DuckLake catalog but does not expose "
            "a qualified Quack state-owner endpoint; direct source-file reads are "
            "admitted only after the LGCVF owner stops"
        ),
    }


@contextlib.contextmanager
def _exclusive_projection_checkpoint(paths: Mapping[str, Path]) -> Any:
    """Hold the controller lock so an owner cannot race a direct checkpoint."""

    lock_path = paths["controller_lock"]
    handle = _open_private_lock(lock_path)
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SuccessorOperatorError(
                "LGCVF owner is active; refusing direct DuckLake checkpoint"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def project_ducklake_once(root: Path = ROOT) -> dict[str, Any]:
    paths = _paths(root)
    with _exclusive_projection_checkpoint(paths):
        return _project_ducklake_once_locked(root)


def _open_projection_plane(root: Path, projection_root: Path) -> Any:
    """Open the stopped projection with a strict local LOAD-only policy."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        open_board_control_plane,
    )

    return open_board_control_plane(
        root,
        root=projection_root,
        allow_extension_install=False,
    )


def _project_ducklake_once_locked(root: Path) -> dict[str, Any]:
    paths = _paths(root)
    preflight = projection_preflight(root, _checkpoint_lock_held=True)
    if preflight.get("valid") is not True:
        raise SuccessorOperatorError("DuckLake projection preflight is not valid")
    if paths["projection_receipt"].exists():
        raise SuccessorOperatorError(
            "refusing to overwrite DuckLake projection receipt"
        )
    provenance = _load_provenance(paths, root=root)
    source_digest = _sha256_regular_file(
        paths["successor_database"],
        noun="successor control database",
        require_private_owner=True,
    )
    import duckdb

    source = duckdb.connect(str(paths["successor_database"]), read_only=True)
    try:
        columns = tuple(
            str(item[0])
            for item in source.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'tasks' ORDER BY ordinal_position"
            ).fetchall()
        )
        rows = source.execute(
            "SELECT * FROM tasks ORDER BY ordinal, task_cid"
        ).fetchall()
    finally:
        source.close()
    tasks: list[dict[str, Any]] = []
    for row in rows:
        record = {columns[index]: row[index] for index in range(len(columns))}
        body: dict[str, Any] = {}
        try:
            parsed = json.loads(str(record.get("body_json") or "{}"))
            if isinstance(parsed, dict):
                body = parsed
        except json.JSONDecodeError:
            pass
        tasks.append(
            {
                "task_id": str(
                    record.get("task_alias") or record.get("task_cid") or ""
                ),
                "status": str(record.get("status") or ""),
                "title": str(body.get("title") or ""),
                "depends_on": body.get("depends_on") or [],
                "body": body,
            }
        )
    with _open_projection_plane(root, paths["projection_root"]) as plane:
        registration = plane.register_board(
            "logic-governed-compositional-verification-fabric-history-shadow-v1",
            source_path=str(paths["successor_database"]),
            source_kind="duckdb-stopped-checkpoint-observation",
            merge_target_branch="agent/logic-governed-compositional-verification-fabric-v1",
            extra={
                "authoritative": False,
                "scheduling_authority": False,
                "completion_authority": False,
                "source_provenance_cid": provenance["receipt_cid"],
            },
            tasks=tasks,
        )
        aggregate = plane.aggregate_boards()
        if plane.backend != "ducklake+quack" or not plane.ducklake_attached:
            raise SuccessorOperatorError(
                "physical BoardControlPlane did not admit DuckLake + Quack"
            )
        backend = plane.backend
        extensions = {
            "quack_loaded": plane.quack_loaded,
            "ducklake_loaded": plane.ducklake_loaded,
            "ducklake_attached": plane.ducklake_attached,
        }
    if _sha256_regular_file(paths["successor_database"]) != source_digest:
        raise SuccessorOperatorError("projection source changed during checkpoint")
    receipt = {
        "schema": PROJECTION_RECEIPT_SCHEMA,
        "issued_at": _utc_now(),
        "projection_root": str(paths["projection_root"]),
        "control_catalog_path": str(paths["projection_root"] / "control.duckdb"),
        "ducklake_catalog_path": str(paths["projection_root"] / "lake.ducklake"),
        "ducklake_data_path": str(paths["projection_root"] / "lake-data"),
        "source_database": str(paths["successor_database"]),
        "source_sha256": source_digest,
        "source_provenance_cid": provenance["receipt_cid"],
        "board_namespace": registration["board_namespace"],
        "task_count": len(tasks),
        "backend": backend,
        "extensions": extensions,
        "aggregate": aggregate,
        "authoritative": False,
        "scheduling_authority": False,
        "completion_authority": False,
        "read_by_scheduler": False,
        "quack_endpoint_served": False,
        "requires_stopped_checkpoint": True,
        "production_authorized": False,
    }
    receipt["receipt_cid"] = _content_id(receipt)
    _atomic_json(paths["projection_receipt"], receipt, replace=False)
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("bootstrap")
    sealed = subparsers.add_parser("bootstrap-sealed-continuity")
    sealed.add_argument("--source-root", type=Path, required=True)
    sealed.add_argument("--control-sha256", required=True)
    sealed.add_argument("--coordination-sha256", required=True)
    sealed.add_argument("--execution-sha256", required=True)
    sealed.add_argument("--bootstrap-sha256", required=True)
    sealed.add_argument("--manifest-sha256", required=True)
    sealed.add_argument("--recovery-receipt-sha256", required=True)
    launch = subparsers.add_parser("launch")
    launch.add_argument(
        "--config", type=Path, default=DEFAULT_SUCCESSOR_CONFIG_RELATIVE
    )
    launch.add_argument("--implement", action="store_true")
    launch.add_argument("--duration-seconds", type=float, default=float("inf"))
    subparsers.add_parser("status")
    stop = subparsers.add_parser("stop")
    stop.add_argument("--timeout-seconds", type=float, default=MAX_STOP_SECONDS)
    subparsers.add_parser("projection-preflight")
    subparsers.add_parser("projection-once")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = Path(args.repo_root).resolve()
    try:
        if args.command == "bootstrap":
            result: Any = bootstrap_successor(root)
        elif args.command == "bootstrap-sealed-continuity":
            result = bootstrap_sealed_successor(
                root=root,
                source_root=Path(args.source_root),
                control_sha256=str(args.control_sha256),
                coordination_sha256=str(args.coordination_sha256),
                execution_sha256=str(args.execution_sha256),
                bootstrap_sha256=str(args.bootstrap_sha256),
                manifest_sha256=str(args.manifest_sha256),
                recovery_receipt_sha256=str(args.recovery_receipt_sha256),
            )
        elif args.command == "launch":
            config = Path(args.config)
            if not config.is_absolute():
                config = _contained(root, config)
            return run_successor(
                config,
                root=root,
                implement=bool(args.implement),
                duration_seconds=float(args.duration_seconds),
            )
        elif args.command == "status":
            result = controller_status(root)
        elif args.command == "stop":
            result = stop_controller(root, timeout_seconds=float(args.timeout_seconds))
        elif args.command == "projection-preflight":
            result = projection_preflight(root)
        elif args.command == "projection-once":
            result = project_ducklake_once(root)
        else:  # pragma: no cover - argparse closes this branch.
            parser.error("unsupported command")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema": CONTROLLER_STATUS_SCHEMA,
                    "valid": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
