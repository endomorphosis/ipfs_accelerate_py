"""Explicit owner-local physical maintenance of one existing execution WAL.

This component grants no authority. A separately admitted native maintenance
owner must hold source/resource custody for the exact existing files. Its
retained control tuple selects the attempt; it is not a live control read or
claim authority. The ordinary read-only status API is not changed. Logical
running/UNKNOWN rows remain running/UNKNOWN: CHECKPOINT does
not settle an attempt, demonstrate non-dispatch, or authorize a retry.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, ClassVar

from . import native_execution_status_owner as status

BINDING_SCHEMA = "ipfs_accelerate_py/agent-supervisor/native-execution-wal-maintenance-binding@1"
RECEIPT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/native-execution-wal-maintenance-receipt@1"
MAX_LIFETIME_SECONDS = 90.0
MAX_INPUT_BYTES = 8 * 1024**2
MAX_SHADOW_BYTES = 16 * 1024**2
MAX_LOGICAL_BYTES = 8 * 1024**2
MAX_ROWS_PER_TABLE = 128
MAX_SCALAR_BYTES = 16_384
_FLAGS = os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_IDENTITY_KEYS = frozenset({"device", "inode", "size", "mtime_ns"})


class NativeExecutionWalMaintenanceError(RuntimeError):
    """Closed reason only; inspect the owner's detached failure_state too."""


def _refuse(reason: str) -> None:
    raise NativeExecutionWalMaintenanceError(reason) from None


def _json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        _refuse("maintenance_json_invalid")


def _identity(value: Any) -> Mapping[str, int]:
    if (not isinstance(value, Mapping) or set(value) != _IDENTITY_KEYS
            or any(type(value[key]) is not int or value[key] < (1 if key == "inode" else 0)
                   or value[key] > 2**63 - 1 for key in _IDENTITY_KEYS)):
        _refuse("maintenance_file_identity_invalid")
    return MappingProxyType(dict(value))


def _path(value: Any) -> Path:
    if type(value) is not str or not value or len(value.encode("utf-8")) > 4096 or "\x00" in value:
        _refuse("maintenance_path_invalid")
    result = Path(value)
    if not result.is_absolute() or str(result) != value or ".." in result.parts:
        _refuse("maintenance_path_invalid")
    return result


@dataclass(frozen=True)
class NativeExecutionWalMaintenanceBinding:
    """Server-selected current file selectors, not a maintenance capability."""

    status_binding: status.NativeExecutionStatusBinding
    wal_file_identity: Mapping[str, int]
    policy_lock_identity: Mapping[str, int]
    backup_directory: str
    maintenance_source_sha256: str
    SCHEMA: ClassVar[str] = BINDING_SCHEMA

    def __post_init__(self) -> None:
        if type(self.status_binding) is not status.NativeExecutionStatusBinding:
            _refuse("maintenance_status_binding_invalid")
        object.__setattr__(self, "wal_file_identity", _identity(self.wal_file_identity))
        object.__setattr__(self, "policy_lock_identity", _identity(self.policy_lock_identity))
        directory = _path(self.backup_directory)
        execution = Path(self.status_binding.execution_path)
        if directory == execution.parent or execution.parent in directory.parents:
            _refuse("maintenance_backup_directory_inside_original_lane")
        if (type(self.maintenance_source_sha256) is not str
                or re.fullmatch(r"[0-9a-f]{64}", self.maintenance_source_sha256) is None):
            _refuse("maintenance_source_pin_invalid")
        if (self.wal_file_identity["size"] <= 0
                or self.status_binding.execution_file_identity["size"] <= 0
                or self.wal_file_identity["size"] + self.status_binding.execution_file_identity["size"] > MAX_INPUT_BYTES):
            _refuse("maintenance_input_bytes_exceeded")
        identities = [self.wal_file_identity, self.policy_lock_identity,
                      self.status_binding.execution_file_identity,
                      self.status_binding.writer_lock_identity]
        if len({(item["device"], item["inode"]) for item in identities}) != 4:
            _refuse("maintenance_original_files_alias")

    @property
    def execution_path(self) -> Path:
        return Path(self.status_binding.execution_path)

    @property
    def wal_path(self) -> Path:
        return Path(str(self.execution_path) + ".wal")

    @property
    def policy_lock_path(self) -> Path:
        return self.execution_path.with_name("." + self.execution_path.name + ".lock")

    @property
    def binding_cid(self) -> str:
        return "sha256:" + hashlib.sha256(_json(self.to_dict())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "status_binding": self.status_binding.to_dict(),
                "wal_file_identity": dict(self.wal_file_identity),
                "policy_lock_identity": dict(self.policy_lock_identity),
                "backup_directory": self.backup_directory,
                "maintenance_source_sha256": self.maintenance_source_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NativeExecutionWalMaintenanceBinding:
        if (not isinstance(value, Mapping)
                or set(value) != {"schema", *(field.name for field in fields(cls))}
                or value.get("schema") != cls.SCHEMA):
            _refuse("maintenance_binding_shape_invalid")
        try:
            selected = status.NativeExecutionStatusBinding.from_dict(value["status_binding"])
        except status.NativeExecutionStatusError:
            _refuse("maintenance_status_binding_invalid")
        return cls(selected, value["wal_file_identity"], value["policy_lock_identity"],
                   value["backup_directory"], value["maintenance_source_sha256"])


class _SnapshotValidator:
    """Reuse the real frozen reader's schema SQL with an owned connection.

    No reader is opened, rebound, monkeypatched or represented as admitted.
    The reused method is a bounded validator, not its filesystem/authority gate.
    """

    def __init__(self, owner: NativeExecutionWalMaintenanceOwner) -> None:
        self.owner = owner
        self.binding = owner.binding.status_binding

    def _rows(self, sql: str, values: list[Any] | None = None, *, maximum: int) -> list[tuple[Any, ...]]:
        return self.owner._rows(sql, values, maximum=maximum)

    def metadata(self) -> dict[str, str]:
        try:
            return status.NativeExecutionStatusOwner._verify_schema_metadata(self)
        except status.NativeExecutionStatusError:
            _refuse("maintenance_execution_schema_or_metadata_invalid")


class NativeExecutionWalMaintenanceOwner:
    """Finite physical replay/CHECKPOINT owner, never a worker or SQL service.

    Backups are retained even on success. The first original RW connection is
    marked as a possible physical effect BEFORE the connector is invoked.
    Failed close retains the genuine handle and all original fences for retry.
    """

    def __init__(self, binding: NativeExecutionWalMaintenanceBinding, *, deadline_monotonic: float,
                 custody_guard: Callable[..., None]) -> None:
        now = time.monotonic()
        if (type(binding) is not NativeExecutionWalMaintenanceBinding
                or type(deadline_monotonic) not in (int, float)
                or not math.isfinite(deadline_monotonic)
                or not now < deadline_monotonic <= now + MAX_LIFETIME_SECONDS
                or not callable(custody_guard)):
            _refuse("maintenance_owner_or_deadline_invalid")
        self._binding = binding
        self._deadline_monotonic = float(deadline_monotonic)
        self._custody_guard = custody_guard
        self._lock = threading.RLock()
        self._cancelled = threading.Event()
        self._custody_lost = threading.Event()
        self._monitor_stop = threading.Event()
        self._monitor: threading.Thread | None = None
        self._timer: threading.Timer | None = None
        self._connection: Any = None
        self._fds: dict[str, int] = {}
        self._backup_fd: int | None = None
        self._backup_identity: tuple[int, int] | None = None
        self._used = False
        self._closed = False
        self._state: dict[str, Any] = {
            "phase": "not_started", "physical_mutation_possible": False,
            "original_checkpoint_verified": False, "connection_close_verified": False,
            "original_locks_released": False, "backup_verified": False,
            "completion_authority": False, "automatic_retry_admitted": False,
            "semantic_admission": False,
        }

    @property
    def binding(self) -> NativeExecutionWalMaintenanceBinding:
        return self._binding

    @property
    def deadline_monotonic(self) -> float:
        return self._deadline_monotonic

    @property
    def failure_state(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state)

    def _check(self, *, force_ack: bool = False) -> None:
        if self._closed:
            _refuse("maintenance_owner_closed")
        if self._custody_lost.is_set():
            _refuse("maintenance_custody_lost")
        if self._cancelled.is_set():
            _refuse("maintenance_cancelled")
        if time.monotonic() >= self.deadline_monotonic:
            _refuse("maintenance_deadline_expired")
        try:
            self._custody_guard(force_ack=force_ack, check_only=False)
        except BaseException:
            self._custody_lost.set()
            self.cancel()
            _refuse("maintenance_custody_lost")

    def _watch_custody(self) -> None:
        # This callback mode MUST perform only cheap local liveness/cancel/
        # deadline checks. Only the execution thread owns native IPC ACKs.
        while not self._monitor_stop.wait(0.25):
            try:
                self._custody_guard(force_ack=False, check_only=True)
            except BaseException:
                self._custody_lost.set()
                self.cancel()
                return

    def cancel(self) -> None:
        self._cancelled.set()
        connection = self._connection
        if connection is not None:
            try:
                connection.interrupt()
            except Exception:
                pass

    def _current(self, path: Path) -> dict[str, int]:
        self._check()
        try:
            return status._stat_identity(path)
        except (OSError, status.NativeExecutionStatusError):
            _refuse("maintenance_original_file_unavailable")

    def _opened(self, fd: int) -> dict[str, int]:
        try:
            return status._opened_identity(fd)
        except (OSError, status.NativeExecutionStatusError):
            _refuse("maintenance_opened_file_invalid")

    def _open_existing(self, key: str, path: Path, expected: Mapping[str, int], *, lock: bool = False) -> None:
        if self._current(path) != dict(expected):
            _refuse("maintenance_original_file_drift")
        fd = os.open(path, os.O_RDONLY | _FLAGS)
        self._fds[key] = fd
        if self._opened(fd) != dict(expected) or self._current(path) != dict(expected):
            _refuse("maintenance_original_file_drift")
        if lock:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError):
                _refuse("maintenance_original_lock_contended")

    def _source_hash(self, path: Path) -> str:
        before = self._current(path)
        if before["size"] > status.MAX_SOURCE_BYTES:
            _refuse("maintenance_source_bytes_exceeded")
        fd = os.open(path, os.O_RDONLY | _FLAGS)
        try:
            if self._opened(fd) != before:
                _refuse("maintenance_source_changed")
            digest = hashlib.sha256()
            offset = 0
            while block := os.read(fd, 65_536):
                self._check()
                offset += len(block)
                if offset > status.MAX_SOURCE_BYTES:
                    _refuse("maintenance_source_bytes_exceeded")
                digest.update(block)
            if offset != before["size"] or self._opened(fd) != before or self._current(path) != before:
                _refuse("maintenance_source_changed")
            return digest.hexdigest()
        finally:
            os.close(fd)

    def _verify_sources(self) -> Any:
        from ..task_sources import duckdb_state
        source_pins = self.binding.status_binding.source_bindings
        reader_path = str(Path(status.__file__).absolute())
        policy_path = str(Path(duckdb_state.__file__).absolute())
        policy_code = getattr(duckdb_state.connect_duckdb_with_policy, "__code__", None)
        if (reader_path not in source_pins or policy_path not in source_pins
                or policy_code is None or str(Path(policy_code.co_filename).absolute()) != policy_path):
            _refuse("maintenance_loaded_reader_or_policy_not_bound")
        for source, expected in source_pins.items():
            if self._source_hash(Path(source)) != expected:
                _refuse("maintenance_source_changed")
        if self._source_hash(Path(__file__).absolute()) != self.binding.maintenance_source_sha256:
            _refuse("maintenance_source_changed")
        return duckdb_state

    def _verify_prewrite_files(self) -> None:
        pairs = (
            ("database", self.binding.execution_path, self.binding.status_binding.execution_file_identity),
            ("wal", self.binding.wal_path, self.binding.wal_file_identity),
            ("writer", self.binding.status_binding.writer_lock_path, self.binding.status_binding.writer_lock_identity),
            ("policy", self.binding.policy_lock_path, self.binding.policy_lock_identity),
        )
        for key, path, expected in pairs:
            if self._current(path) != dict(expected) or self._opened(self._fds[key]) != dict(expected):
                _refuse("maintenance_original_file_drift")
        for suffix in (".wal.checkpoint", ".wal.recovery"):
            if os.path.lexists(str(self.binding.execution_path) + suffix):
                _refuse("maintenance_unknown_recovery_artifact")

    def _hash_fd(self, fd: int, *, maximum: int) -> str:
        before = self._opened(fd)
        if before["size"] > maximum:
            _refuse("maintenance_file_bytes_exceeded")
        digest = hashlib.sha256()
        offset = 0
        while offset < before["size"]:
            self._check()
            block = os.pread(fd, min(65_536, before["size"] - offset), offset)
            if not block:
                _refuse("maintenance_file_shortened")
            digest.update(block)
            offset += len(block)
        if self._opened(fd) != before:
            _refuse("maintenance_file_changed_while_hashed")
        return digest.hexdigest()

    def _require_backup_directory(self, *, empty: bool = False) -> None:
        path = Path(self.binding.backup_directory)
        for part in (path, *path.parents):
            if stat.S_ISLNK(part.lstat().st_mode):
                _refuse("maintenance_backup_directory_aliased")
        info = path.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            _refuse("maintenance_backup_directory_not_private")
        if self._backup_identity is not None and (info.st_dev, info.st_ino) != self._backup_identity:
            _refuse("maintenance_backup_directory_replaced")
        if self._backup_fd is not None:
            opened = os.fstat(self._backup_fd)
            if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                _refuse("maintenance_backup_directory_replaced")
        if empty and os.listdir(self._backup_fd) != []:
            _refuse("maintenance_backup_directory_not_empty")

    def _copy_fd(self, source: int, *, directory: int, name: str, maximum: int) -> tuple[dict[str, int], str]:
        before = self._opened(source)
        if before["size"] <= 0 or before["size"] > maximum:
            _refuse("maintenance_copy_bytes_exceeded")
        target = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FLAGS, 0o600, dir_fd=directory)
        digest = hashlib.sha256()
        offset = 0
        try:
            while offset < before["size"]:
                self._check()
                block = os.pread(source, min(65_536, before["size"] - offset), offset)
                if not block:
                    _refuse("maintenance_copy_source_shortened")
                digest.update(block)
                written = 0
                while written < len(block):
                    self._check()
                    count = os.write(target, block[written:])
                    if count <= 0:
                        _refuse("maintenance_copy_write_failed")
                    written += count
                offset += len(block)
            os.fsync(target)
            if self._opened(source) != before:
                _refuse("maintenance_copy_source_changed")
            result = self._opened(target)
            if result["size"] != before["size"] or stat.S_IMODE(os.fstat(target).st_mode) != 0o600:
                _refuse("maintenance_backup_copy_invalid")
        finally:
            os.close(target)
        os.fsync(directory)
        return result, digest.hexdigest()

    def _verify_backup(self, name: str, expected: Mapping[str, int], digest: str) -> None:
        self._require_backup_directory()
        fd = os.open(name, os.O_RDONLY | _FLAGS, dir_fd=self._backup_fd)
        try:
            if self._opened(fd) != dict(expected) or self._hash_fd(fd, maximum=MAX_INPUT_BYTES) != digest:
                _refuse("maintenance_retained_backup_changed")
            named = os.stat(name, dir_fd=self._backup_fd, follow_symlinks=False)
            if (named.st_dev, named.st_ino) != (expected["device"], expected["inode"]):
                _refuse("maintenance_retained_backup_changed")
        finally:
            os.close(fd)

    def _rows(self, sql: str, values: list[Any] | None = None, *, maximum: int) -> list[tuple[Any, ...]]:
        self._check()
        try:
            rows = self._connection.execute(sql, values or []).fetchmany(maximum + 1)
        except Exception:
            self._check()
            _refuse("maintenance_bounded_query_failed")
        self._check()
        if len(rows) > maximum or any(type(row) is not tuple for row in rows):
            _refuse("maintenance_result_rows_exceeded")
        return rows

    def _snapshot(self) -> dict[str, Any]:
        self._check()
        self._connection.execute("BEGIN TRANSACTION")
        try:
            tables = self._rows("""
                SELECT CASE WHEN octet_length(encode(table_schema))<=128 THEN table_schema ELSE NULL END,
                       CASE WHEN octet_length(encode(table_name))<=128 THEN table_name ELSE NULL END,
                       CASE WHEN octet_length(encode(table_type))<=32 THEN table_type ELSE NULL END
                FROM information_schema.tables WHERE table_catalog=current_database()
                  AND table_schema NOT IN ('information_schema','pg_catalog')
                ORDER BY table_schema,table_name LIMIT 8
            """, maximum=6)
            if tables != [("main", table, "BASE TABLE") for table in sorted(status._COLUMNS)]:
                _refuse("maintenance_extra_or_missing_user_relation")
            metadata = _SnapshotValidator(self).metadata()
            if set(metadata) != status._METADATA_KEYS:
                _refuse("maintenance_metadata_population_changed")
            result: dict[str, Any] = {"metadata": metadata, "tables": {}}
            total = len(_json(metadata))
            for table, columns in status._COLUMNS.items():
                expressions = [
                    (f'CASE WHEN octet_length(encode("{column}"))<={MAX_SCALAR_BYTES} '
                     f'THEN "{column}" ELSE NULL END') if kind == "VARCHAR" else f'"{column}"'
                    for column, kind in columns
                ]
                order = ",".join(str(index + 1) for index in range(len(columns)))
                rows = self._rows(f'SELECT {",".join(expressions)} FROM main."{table}" '
                                  f'ORDER BY {order} LIMIT {MAX_ROWS_PER_TABLE + 1}', maximum=MAX_ROWS_PER_TABLE)
                for row in rows:
                    if len(row) != len(columns):
                        _refuse("maintenance_row_shape_invalid")
                    for (column, kind), value in zip(columns, row):
                        if kind == "VARCHAR" and (type(value) is not str or len(value.encode("utf-8")) > MAX_SCALAR_BYTES):
                            _refuse("maintenance_scalar_bytes_exceeded")
                        if kind == "BIGINT" and value is not None and type(value) is not int:
                            _refuse("maintenance_scalar_type_invalid")
                        if value is None and (table, column) != ("database_task_attempts", "finished_at_ms"):
                            _refuse("maintenance_null_or_oversized_scalar")
                    total += len(_json(row))
                    if total > MAX_LOGICAL_BYTES:
                        _refuse("maintenance_logical_bytes_exceeded")
                result["tables"][table] = rows
            attempts = [dict(zip((name for name, _kind in status._COLUMNS["database_task_attempts"]), row))
                        for row in result["tables"]["database_task_attempts"]
                        if row[0] == self.binding.status_binding.attempt_id]
            if len(attempts) != 1:
                _refuse("maintenance_exact_attempt_missing")
            attempt = attempts[0]
            for key in ("attempt_id", "claim_id", "task_cid", "attempt_number", "owner_session_id",
                        "fencing_token", "fence_epoch", "lease_id"):
                expected = getattr(self.binding.status_binding, key)
                if type(attempt[key]) is not type(expected) or attempt[key] != expected:
                    _refuse("maintenance_attempt_join_mismatch")
            if attempt["status"] not in {"running", "succeeded", "failed", "blocked"} or attempt["committed_phase"] not in status._PHASES:
                _refuse("maintenance_attempt_cursor_invalid")
            result["logical_bytes"] = total
            result["logical_sha256"] = hashlib.sha256(_json(result)).hexdigest()
        except BaseException:
            try:
                self._connection.execute("ROLLBACK")
            except Exception:
                pass
            raise
        else:
            try:
                self._connection.execute("ROLLBACK")
            except Exception:
                _refuse("maintenance_snapshot_release_failed")
            return result

    def _connect(self, path: Path, *, read_only: bool) -> None:
        self._check()
        policy = self._verify_sources()
        import duckdb
        owner = self

        class RetainedBackend:
            """Local capture only; the genuine policy and module stay intact."""

            @staticmethod
            def connect(*args: Any, **kwargs: Any) -> Any:
                owner._check()
                raw = duckdb.connect(*args, **kwargs)
                owner._connection = raw
                return raw

        if path == self.binding.execution_path and not read_only:
            self._verify_prewrite_files()
            self._check(force_ack=True)
            self._state["physical_mutation_possible"] = True
        returned = policy.connect_duckdb_with_policy(
            RetainedBackend, path, read_only=read_only,
            configuration={"threads": 1, "memory_limit": "256MB"},
        )
        if returned is not self._connection:
            _refuse("maintenance_backend_handle_capture_failed")
        self._check()

    def _close_connection(self) -> None:
        if self._connection is not None:
            # Do not clear custody before the actual native close succeeds.
            self._connection.close()
            self._connection = None

    def _after_original_checkpoint(self) -> dict[str, int]:
        current = self._current(self.binding.execution_path)
        expected = self.binding.status_binding.execution_file_identity
        if (current["device"], current["inode"]) != (expected["device"], expected["inode"]):
            _refuse("maintenance_checkpoint_database_replaced")
        if self._opened(self._fds["database"]) != current or current["size"] > MAX_SHADOW_BYTES:
            _refuse("maintenance_checkpoint_database_invalid")
        if os.path.lexists(self.binding.wal_path):
            _refuse("maintenance_original_wal_remains")
        for key, path, expected in (
            ("writer", self.binding.status_binding.writer_lock_path, self.binding.status_binding.writer_lock_identity),
            ("policy", self.binding.policy_lock_path, self.binding.policy_lock_identity),
        ):
            if self._current(path) != dict(expected) or self._opened(self._fds[key]) != dict(expected):
                _refuse("maintenance_original_lock_replaced")
        return current

    def run(self) -> dict[str, Any]:
        with self._lock:
            self._check()
            if self._used:
                _refuse("maintenance_is_one_shot")
            self._used = True
            receipt: dict[str, Any] | None = None
            try:
                self._monitor = threading.Thread(target=self._watch_custody, daemon=True,
                                                 name="native-execution-maintenance-custody")
                self._monitor.start()
                self._verify_sources()
                self._open_existing("writer", self.binding.status_binding.writer_lock_path,
                                    self.binding.status_binding.writer_lock_identity, lock=True)
                self._open_existing("policy", self.binding.policy_lock_path, self.binding.policy_lock_identity, lock=True)
                self._open_existing("database", self.binding.execution_path, self.binding.status_binding.execution_file_identity)
                self._open_existing("wal", self.binding.wal_path, self.binding.wal_file_identity)
                self._verify_prewrite_files()
                self._require_backup_directory()
                self._backup_fd = os.open(self.binding.backup_directory, os.O_RDONLY | os.O_DIRECTORY | _FLAGS)
                info = os.fstat(self._backup_fd)
                self._backup_identity = (info.st_dev, info.st_ino)
                self._require_backup_directory(empty=True)
                parent_fd = os.open(Path(self.binding.backup_directory).parent,
                                    os.O_RDONLY | os.O_DIRECTORY | _FLAGS)
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
                self._timer = threading.Timer(max(0.0, self.deadline_monotonic - time.monotonic()), self.cancel)
                self._timer.daemon = True
                self._timer.start()
                self._state["phase"] = "backing_up_originals"
                backups = {}
                for key, name in (("database", "original.db"), ("wal", "original.db.wal")):
                    file_identity, digest = self._copy_fd(self._fds[key], directory=self._backup_fd,
                                                          name=name, maximum=MAX_INPUT_BYTES)
                    backups[key] = {"name": name, "identity": file_identity, "sha256": digest}
                    self._verify_backup(name, file_identity, digest)
                self._verify_prewrite_files()
                self._state["backup_verified"] = True
                self._state["backup_sha256"] = {key: item["sha256"] for key, item in backups.items()}
                os.mkdir("shadow", mode=0o700, dir_fd=self._backup_fd)
                shadow_fd = os.open("shadow", os.O_RDONLY | os.O_DIRECTORY | _FLAGS, dir_fd=self._backup_fd)
                try:
                    for key, name in (("database", "shadow.db"), ("wal", "shadow.db.wal")):
                        source_fd = os.open(backups[key]["name"], os.O_RDONLY | _FLAGS, dir_fd=self._backup_fd)
                        try:
                            copied, copied_hash = self._copy_fd(source_fd, directory=shadow_fd, name=name, maximum=MAX_INPUT_BYTES)
                            if copied_hash != backups[key]["sha256"]:
                                _refuse("maintenance_shadow_copy_changed")
                            verify_fd = os.open(name, os.O_RDONLY | _FLAGS, dir_fd=shadow_fd)
                            try:
                                if self._opened(verify_fd) != copied or self._hash_fd(verify_fd, maximum=MAX_INPUT_BYTES) != copied_hash:
                                    _refuse("maintenance_shadow_copy_changed")
                            finally:
                                os.close(verify_fd)
                        finally:
                            os.close(source_fd)
                    os.fsync(shadow_fd)
                finally:
                    os.close(shadow_fd)
                os.fsync(self._backup_fd)
                shadow_path = Path(self.binding.backup_directory) / "shadow" / "shadow.db"
                self._state["phase"] = "shadow_replay"
                self._connect(shadow_path, read_only=False)
                self._check()
                self._connection.execute("CHECKPOINT")
                baseline = self._snapshot()
                self._close_connection()
                if os.path.lexists(str(shadow_path) + ".wal") or self._current(shadow_path)["size"] > MAX_SHADOW_BYTES:
                    _refuse("maintenance_shadow_replay_incomplete")
                self._state["shadow_preflight_verified"] = True
                self._verify_sources()
                self._verify_prewrite_files()
                for key, backup in backups.items():
                    self._verify_backup(backup["name"], backup["identity"], backup["sha256"])
                    if self._hash_fd(self._fds[key], maximum=MAX_INPUT_BYTES) != backup["sha256"]:
                        _refuse("maintenance_original_hash_changed_before_write")
                self._check()
                self._state["phase"] = "original_replay"
                self._connect(self.binding.execution_path, read_only=False)
                self._check()
                self._connection.execute("CHECKPOINT")
                if self._snapshot() != baseline:
                    _refuse("maintenance_original_logical_state_changed")
                self._close_connection()
                post_identity = self._after_original_checkpoint()
                self._state["phase"] = "original_readonly_verification"
                self._check(force_ack=True)
                self._connect(self.binding.execution_path, read_only=True)
                if self._snapshot() != baseline:
                    _refuse("maintenance_readonly_logical_state_changed")
                self._check(force_ack=True)
                self._close_connection()
                if self._after_original_checkpoint() != post_identity:
                    _refuse("maintenance_original_changed_after_readonly")
                os.fsync(self._fds["database"])
                original_parent_fd = os.open(self.binding.execution_path.parent,
                                             os.O_RDONLY | os.O_DIRECTORY | _FLAGS)
                try:
                    os.fsync(original_parent_fd)
                finally:
                    os.close(original_parent_fd)
                if self._after_original_checkpoint() != post_identity:
                    _refuse("maintenance_original_changed_during_durable_close")
                self._verify_sources()
                for backup in backups.values():
                    self._verify_backup(backup["name"], backup["identity"], backup["sha256"])
                self._state["original_checkpoint_verified"] = True
                self._state["phase"] = "verified_pending_close"
                receipt = {
                    "schema": RECEIPT_SCHEMA, "binding_cid": self.binding.binding_cid,
                    "physical_maintenance_only": True, "checkpoint_executed": True,
                    "physical_mutation_possible": True, "logical_rows_preserved": True,
                    "all_metadata_preserved": True, "observed_process_instance_id": baseline["metadata"]["process_instance_id"],
                    "original_process_impersonated": False, "native_store_uuid": None,
                    "historical_inode_attestation": False, "logical_sha256": baseline["logical_sha256"],
                    "table_row_counts": {key: len(rows) for key, rows in baseline["tables"].items()},
                    "logical_bytes": baseline["logical_bytes"], "execution_file_identity": post_identity,
                    "wal_after": "absent", "retained_backups": copy.deepcopy(backups),
                    "backup_directory": self.binding.backup_directory,
                    "provider_effect_state": "unknown", "unknown_not_settled": True,
                    "coordination_authority_observed": False, "control_join_verified_by_component": False,
                    "completion_authority": False, "automatic_retry_admitted": False, "semantic_admission": False,
                }
            except BaseException as exc:
                self._state["failure_reason"] = str(exc) if type(exc) is NativeExecutionWalMaintenanceError else "maintenance_operation_failed"
                if type(exc) is NativeExecutionWalMaintenanceError:
                    raise
                _refuse("maintenance_operation_failed")
            finally:
                self.close()
            if receipt is None:
                _refuse("maintenance_receipt_missing")
            receipt["connection_close_verified"] = self._state["connection_close_verified"]
            receipt["original_locks_released"] = self._state["original_locks_released"]
            receipt["receipt_cid"] = "sha256:" + hashlib.sha256(_json(receipt)).hexdigest()
            return copy.deepcopy(receipt)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._monitor_stop.set()
            monitor = self._monitor
            if monitor is not None and monitor is not threading.current_thread():
                if monitor.ident is not None:
                    monitor.join(timeout=0.5)
                if monitor.is_alive():
                    self._state["close_retry_required"] = True
                    _refuse("maintenance_custody_watch_not_quiescent")
                self._monitor = None
            try:
                self._close_connection()
            except BaseException:
                self._state["connection_close_verified"] = False
                self._state["close_retry_required"] = True
                _refuse("maintenance_native_close_failed_custody_retained")
            self._state["connection_close_verified"] = True
            for key in ("wal", "database", "policy", "writer"):
                fd = self._fds.get(key)
                if fd is not None:
                    os.close(fd)
                    del self._fds[key]
            if self._backup_fd is not None:
                os.close(self._backup_fd)
                self._backup_fd = None
            self._state["original_locks_released"] = True
            self._state["close_retry_required"] = False
            if self._state["original_checkpoint_verified"]:
                self._state["phase"] = "closed_verified"


__all__ = ["NativeExecutionWalMaintenanceBinding", "NativeExecutionWalMaintenanceOwner",
           "NativeExecutionWalMaintenanceError"]
