#!/usr/bin/env python3
"""Run the additive LGCVF DuckDB + Quack successor controller.

The canonical run-v16 database is forensic input and the sealed run-v17
configuration remains an embedded, single-writer recovery target.  This
operator therefore has two explicit stages:

* ``bootstrap`` verifies the canonical run-v17 recovery and publishes one
  no-overwrite run-v18 database clone with a provenance receipt;
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
    "ipfs_accelerate_py/agent-supervisor/lgcvf-quack-successor-provenance@1"
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
MAX_DATABASE_BYTES: Final = 8 * 1024 * 1024 * 1024
MAX_JSON_BYTES: Final = 4 * 1024 * 1024
MAX_SECRET_SURFACE_BYTES: Final = 1024 * 1024 * 1024
MAX_STOP_SECONDS: Final = 20.0
UNIX_SOCKET_PATH_CEILING: Final = 100


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
        raise SuccessorOperatorError(f"runtime path escapes repository: {relative}") from exc
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
        raise SuccessorOperatorError("derived state-owner socket path exceeds its bound")
    paths["owner_socket"] = owner_socket
    return paths


def _strict_json(path: Path, *, expected_schema: str = "") -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SuccessorOperatorError(f"required receipt is unreadable: {path}") from exc
    if len(raw) > MAX_JSON_BYTES:
        raise SuccessorOperatorError(f"receipt exceeds its byte bound: {path}")
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


def _sha256_regular_file(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SuccessorOperatorError(f"database is unreadable: {path}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > MAX_DATABASE_BYTES
        ):
            raise SuccessorOperatorError(f"database is not a bounded regular file: {path}")
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
            raise SuccessorOperatorError(f"database changed while hashing: {path}")
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
        raise SuccessorOperatorError(f"could not inspect credential surface: {path}") from exc
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
    """Publish one verified profile clone without replacing any existing byte."""

    source = Path(source_database).resolve()
    target = Path(target_database).resolve()
    provenance = Path(provenance_path).resolve()
    if source.parent.name != "run-v17" or target.parent.name != "run-v18":
        raise SuccessorOperatorError("successor clone must be run-v17 -> run-v18")
    if source == target or target.exists() or provenance.exists():
        raise SuccessorOperatorError("refusing to overwrite an existing successor")
    if source.with_name(source.name + ".wal").exists():
        raise SuccessorOperatorError("run-v17 control database has a live WAL")
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
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = target.parent / f".{target.stem}.{uuid.uuid4().hex}.tmp.duckdb"
    source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    target_descriptor: int | None = None
    copy_complete = False
    try:
        source_before = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_before.st_mode) or source_before.st_size > MAX_DATABASE_BYTES:
            raise SuccessorOperatorError("run-v17 source is not a bounded regular database")
        target_descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
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
        source_after = os.fstat(source_descriptor)
        if (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_size,
            source_before.st_mtime_ns,
            source_before.st_ctime_ns,
        ) != (
            source_after.st_dev,
            source_after.st_ino,
            source_after.st_size,
            source_after.st_mtime_ns,
            source_after.st_ctime_ns,
        ):
            raise SuccessorOperatorError("run-v17 changed during clone")
        copy_complete = True
    finally:
        os.close(source_descriptor)
        if target_descriptor is not None:
            os.close(target_descriptor)
        if not copy_complete:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    try:
        target_verification = _verify_profile(temporary)
        target_identity = _database_identity(temporary)
        target_digest = _sha256_regular_file(temporary)
        if (
            _sha256_regular_file(source) != source_digest
            or target_digest != source_digest
            or target_identity != source_identity
            or target_verification.get("schema_fingerprint")
            != source_verification.get("schema_fingerprint")
        ):
            raise SuccessorOperatorError("run-v18 clone differs from verified run-v17")
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise SuccessorOperatorError("run-v18 target appeared during clone") from exc
        temporary.unlink()
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

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
        "clone_preserves_database_uuid": True,
        "owner_generation_rotates_on_start": True,
        "source_database_statuses_read": False,
        "authoritative_for_release": False,
        "production_authorized": False,
    }
    receipt["receipt_cid"] = _content_id(receipt)
    _atomic_json(provenance, receipt, replace=False)
    if _sha256_regular_file(source) != source_digest:
        raise SuccessorOperatorError("run-v17 changed after successor publication")
    return receipt


def bootstrap_successor(root: Path = ROOT) -> dict[str, Any]:
    paths = _paths(root)
    ignored = subprocess.run(
        [
            "git",
            "check-ignore",
            "-q",
            "--no-index",
            str(SUCCESSOR_DATABASE_RELATIVE),
        ],
        cwd=root,
        check=False,
    )
    if ignored.returncode != 0:
        raise SuccessorOperatorError("run-v18 successor root is not Git-ignored")
    recovery = _canonical_recovery_verification(root)
    return clone_verified_successor(
        paths["source_database"],
        paths["successor_database"],
        paths["provenance"],
        recovery_verification=recovery,
    )


def _load_provenance(paths: Mapping[str, Path]) -> dict[str, Any]:
    receipt = _strict_json(paths["provenance"], expected_schema=PROVENANCE_SCHEMA)
    database = paths["successor_database"]
    if (
        receipt.get("source_database") != str(paths["source_database"])
        or receipt.get("target_database") != str(database)
        or _sha256_regular_file(paths["source_database"])
        != receipt.get("source_sha256")
    ):
        raise SuccessorOperatorError("successor provenance no longer binds run-v17")
    verification = _verify_profile(database)
    identity = _database_identity(database)
    if (
        verification.get("schema_fingerprint") != receipt.get("schema_fingerprint")
        or verification.get("catalog_fingerprint") != receipt.get("catalog_fingerprint")
        or identity.get("database_uuid") != receipt.get("database_uuid")
    ):
        raise SuccessorOperatorError("successor database identity differs from provenance")
    return receipt


def _parse_quack_endpoint(endpoint: str) -> tuple[str, int]:
    match = re.fullmatch(r"quack:(?://)?(127\.0\.0\.1|localhost):(\d{1,5})", endpoint)
    if match is None or not 1 <= int(match.group(2)) <= 65535:
        raise SuccessorOperatorError("successor Quack endpoint must be fixed loopback")
    return match.group(1), int(match.group(2))


def _validate_successor_board(config_path: Path, root: Path = ROOT) -> tuple[Any, Any, str, int]:
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
        or raw_program.get("schema_profile")
        != "datasets-authoritative-operational"
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
        raise SuccessorOperatorError("scheduler config is not the exact four-lane successor")
    host, port = _parse_quack_endpoint(program.quack_endpoint)
    preflight = preflight_configured_board(board)
    if preflight.get("valid") is not True:
        raise SuccessorOperatorError(
            "configured-board preflight failed: " + ", ".join(preflight.get("errors") or ())
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
        raise SuccessorOperatorError("state-owner socket directory is unavailable") from exc
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
    provenance = _load_provenance(paths)
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
            raise SuccessorOperatorError("another successor controller owns the lock") from exc

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
            raise SuccessorOperatorError("owner endpoint differs from scheduler program")
        if server._vault is None:
            stop_owner()
            raise SuccessorOperatorError("owner token vault is unavailable")
        token = server._vault.resolve(identity.secret_handle)
        # Harden without copying the credential into the controller environment.
        harden_state_authority_process({TOKEN_ENV: token})
        token_path = paths["owner_state"] / (
            identity.secret_handle.replace(":", "_").replace("/", "_")
            + ".quack-token"
        )
        if token_path.exists() or token.encode("ascii") in _canonical_bytes(server.status()):
            stop_owner()
            raise SuccessorOperatorError("owner published its Quack attach token")

        command = [
            sys.executable,
            str(_contained(root, "scripts/ops/agent_supervisor/configured_board_scheduler.py")),
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
                raise SuccessorOperatorError("mutation inbox pump failed: " + pump_error)
        finally:
            for signum, handler in prior_handlers.items():
                signal.signal(signum, handler)
            if scheduler is not None and scheduler.poll() is None and scheduler_birth is not None:
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
                scheduler_returncode=(scheduler.returncode if scheduler is not None else None),
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
    status = _strict_json(paths["controller_status"], expected_schema=CONTROLLER_STATUS_SCHEMA)
    birth = ProcessBirthIdentity.from_dict(status.get("controller_birth"))
    observed = owner_liveness(birth)
    projection = dict(status.get("ducklake_projection") or {})
    projection["receipt_present"] = paths["projection_receipt"].is_file()
    return {
        **status,
        "observed_controller_liveness": observed.value,
        "running": observed is OwnerLiveness.ALIVE and status.get("lifecycle") == "ready",
        "ducklake_projection": projection,
    }


def stop_controller(root: Path = ROOT, *, timeout_seconds: float = MAX_STOP_SECONDS) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        ProcessBirthIdentity,
    )

    status = controller_status(root)
    birth = ProcessBirthIdentity.from_dict(status.get("controller_birth"))
    disposition = _terminate_exact(birth, grace_seconds=min(timeout_seconds, 15.0))
    return {"stopped": True, "disposition": disposition, "controller_birth": birth.to_dict()}


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
            _load_provenance(paths)
            source_admitted = True
        except (OSError, RuntimeError, ValueError) as exc:
            source_error = f"{type(exc).__name__}: {exc}"
    return {
        "schema": PROJECTION_RECEIPT_SCHEMA,
        "valid": (
            capability.get("available") is True
            and not running
            and source_admitted
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
        raise SuccessorOperatorError("refusing to overwrite DuckLake projection receipt")
    provenance = _load_provenance(paths)
    source_digest = _sha256_regular_file(paths["successor_database"])
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
        rows = source.execute("SELECT * FROM tasks ORDER BY ordinal, task_cid").fetchall()
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
                "task_id": str(record.get("task_alias") or record.get("task_cid") or ""),
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
    launch = subparsers.add_parser("launch")
    launch.add_argument("--config", type=Path, default=DEFAULT_SUCCESSOR_CONFIG_RELATIVE)
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
