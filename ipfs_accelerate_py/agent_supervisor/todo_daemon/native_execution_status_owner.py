"""Bounded observation owner for an existing native execution sidecar.

This is an owner-local component, not a client SQL API or an authorization
factory. The typed owner must authenticate a separate, exact status grant.
Opening this component never restores a worker, rewrites its identity, creates
schema, renews claims or decides whether an interrupted provider may retry.
Only the existing DuckDB execution@1 backend is supported.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

BINDING_SCHEMA = "ipfs_accelerate_py/agent-supervisor/native-execution-status-binding@1"
OBSERVATION_SCHEMA = "ipfs_accelerate_py/agent-supervisor/native-execution-status-observation@1"
EXECUTION_INTERFACE = "DatabaseImplementationDaemon@1"
EXECUTION_SCHEMA = "ipfs_accelerate_py/agent-supervisor/database-implementation-daemon@1"
MAX_LIFETIME_SECONDS = 90.0
MAX_JSON_BYTES = 16_384
MAX_OBSERVATION_BYTES = 262_144
MAX_PHASES = 8
MAX_IDEMPOTENCY_RECORDS = 8
MAX_METADATA_ROWS = 32
MAX_SOURCE_BYTES = 8_388_608
_IDENTITY_KEYS = frozenset({"device", "inode", "size", "mtime_ns"})
_METADATA_KEYS = frozenset({
    "interface", "schema", "authority_mode", "logical_owner_session_id",
    "process_instance_id", "state_schema_revision", "control_schema_profile_id",
    "control_schema_fingerprint", "execution_state_dir", "execution_state_prefix",
    "task_shard_count", "task_shard_index", "strict_task_sharding",
})
_REQUIRED_EXPECTED_METADATA = _METADATA_KEYS - {
    "process_instance_id", "state_schema_revision", "control_schema_profile_id",
    "control_schema_fingerprint",
}
_PHASES = frozenset({"claimed", "context", "provider", "effect", "validation", "complete", "failed", "blocked"})
_ATTEMPT_COLUMNS = (
    "attempt_id", "claim_id", "task_cid", "task_alias", "attempt_number",
    "owner_session_id", "fencing_token", "fence_epoch", "lease_id",
    "committed_phase", "status", "started_at_ms", "finished_at_ms", "revision",
    "body_json",
)
# Exact existing producer column order/types; defaults are not installed here.
_COLUMNS = {
    "daemon_execution_metadata": (("key", "VARCHAR"), ("value", "VARCHAR")),
    "database_task_attempts": tuple((name, "BIGINT" if name in {
        "attempt_number", "fencing_token", "fence_epoch", "started_at_ms",
        "finished_at_ms", "revision",
    } else "VARCHAR") for name in _ATTEMPT_COLUMNS),
    "attempt_phases": (("attempt_id", "VARCHAR"), ("phase", "VARCHAR"),
        ("committed_at_ms", "BIGINT"), ("fencing_token", "BIGINT"),
        ("fence_epoch", "BIGINT"), ("revision", "BIGINT"), ("body_json", "VARCHAR")),
    "provider_invocations": (("invocation_id", "VARCHAR"), ("attempt_id", "VARCHAR"),
        ("task_cid", "VARCHAR"), ("idempotency_key", "VARCHAR"),
        ("owner_session_id", "VARCHAR"), ("recorded_at_ms", "BIGINT"), ("result_json", "VARCHAR")),
    "effect_claims": (("effect_id", "VARCHAR"), ("attempt_id", "VARCHAR"),
        ("task_cid", "VARCHAR"), ("effect_key", "VARCHAR"), ("idempotency_key", "VARCHAR"),
        ("owner_session_id", "VARCHAR"), ("recorded_at_ms", "BIGINT"), ("result_json", "VARCHAR")),
    "daemon_execution_events": (("event_id", "VARCHAR"), ("attempt_id", "VARCHAR"),
        ("task_cid", "VARCHAR"), ("event_type", "VARCHAR"),
        ("recorded_at_ms", "BIGINT"), ("body_json", "VARCHAR")),
}
_CONSTRAINTS = {
    "daemon_execution_metadata": {("PRIMARY KEY", ("key",))},
    "database_task_attempts": {("PRIMARY KEY", ("attempt_id",))},
    "attempt_phases": {("PRIMARY KEY", ("attempt_id", "phase"))},
    "provider_invocations": {("PRIMARY KEY", ("invocation_id",)), ("UNIQUE", ("attempt_id", "idempotency_key"))},
    "effect_claims": {("PRIMARY KEY", ("effect_id",)), ("UNIQUE", ("attempt_id", "idempotency_key"))},
    "daemon_execution_events": {("PRIMARY KEY", ("event_id",))},
}


class NativeExecutionStatusError(RuntimeError):
    """Closed reason code; private row values are never placed in this error."""


def _refuse(reason: str) -> None:
    raise NativeExecutionStatusError(reason) from None


def _text(value: Any, *, empty: bool = False, maximum: int = 4096) -> str:
    if type(value) is not str or (not empty and not value) or len(value.encode("utf-8")) > maximum or "\x00" in value:
        _refuse("invalid_binding_text")
    return value


def _integer(value: Any, *, minimum: int = 1) -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        _refuse("invalid_binding_integer")
    return value


def _path(value: Any) -> Path:
    raw = _text(value)
    path = Path(raw)
    if not path.is_absolute() or str(path) != raw or ".." in path.parts:
        _refuse("invalid_original_path")
    return path


def _identity(value: Any) -> Mapping[str, int]:
    if not isinstance(value, Mapping) or set(value) != _IDENTITY_KEYS:
        _refuse("invalid_file_identity")
    result = {key: _integer(value[key], minimum=0 if key in {"device", "size", "mtime_ns"} else 1) for key in _IDENTITY_KEYS}
    return MappingProxyType(result)


def _stat_identity(path: Path) -> dict[str, int]:
    # Refuse aliases in every existing path component; do not resolve them.
    for part in (path, *path.parents):
        if stat.S_ISLNK(part.lstat().st_mode):
            _refuse("aliased_original_path")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.geteuid():
        _refuse("invalid_original_file")
    return dict(device=info.st_dev, inode=info.st_ino, size=info.st_size, mtime_ns=info.st_mtime_ns)


def _opened_identity(fd: int) -> dict[str, int]:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.geteuid():
        _refuse("invalid_opened_original_file")
    return dict(device=info.st_dev, inode=info.st_ino, size=info.st_size, mtime_ns=info.st_mtime_ns)


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError):
        _refuse("invalid_json")


def _json_object(raw: Any) -> dict[str, Any]:
    if type(raw) is not str or len(raw.encode("utf-8")) > MAX_JSON_BYTES:
        _refuse("json_bound_exceeded")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _refuse("duplicate_json_key")
            result[key] = value
        return result

    def invalid_constant(_value: str) -> None:
        _refuse("nonfinite_json")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (ValueError, TypeError, RecursionError):
        _refuse("malformed_json")
    if type(value) is not dict:
        _refuse("json_not_object")
    # A JSON exponent can produce infinity without parse_constant being called.
    _json_bytes(value)
    return value


@dataclass(frozen=True)
class NativeExecutionStatusBinding:
    """Server-selected immutable input metadata; this object grants no authority.

    execution@1 has no UUID. Physical identity plus existing native identity
    metadata is explicit; ``control_store_id`` identifies ONLY the primary
    control store. Optional expected metadata pins must match if supplied.
    """

    backend: str
    execution_path: str
    execution_file_identity: Mapping[str, int]
    writer_lock_identity: Mapping[str, int]
    expected_metadata: Mapping[str, str]
    source_bindings: Mapping[str, str]
    control_store_id: str
    control_schema_version: int
    task_cid: str
    control_revision: int
    expected_control_status: str
    attempt_id: str
    attempt_number: int
    claim_id: str
    lease_id: str
    owner_session_id: str
    fencing_token: int
    fence_epoch: int

    SCHEMA: ClassVar[str] = BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.backend != "duckdb" or type(self.backend) is not str:
            _refuse("unsupported_execution_backend")
        path = _path(self.execution_path)
        object.__setattr__(self, "execution_file_identity", _identity(self.execution_file_identity))
        object.__setattr__(self, "writer_lock_identity", _identity(self.writer_lock_identity))
        for name in ("control_store_id", "task_cid", "attempt_id", "claim_id", "lease_id", "owner_session_id"):
            _text(getattr(self, name), maximum=1024)
        for name in ("control_schema_version", "control_revision", "attempt_number", "fencing_token", "fence_epoch"):
            _integer(getattr(self, name))
        _text(self.expected_control_status, maximum=32)
        if self.expected_control_status not in {"ready", "todo", "in_progress", "retrying", "blocked", "completed", "failed", "cancelled"}:
            _refuse("invalid_control_status")
        if not isinstance(self.expected_metadata, Mapping) or not _REQUIRED_EXPECTED_METADATA <= set(self.expected_metadata) <= _METADATA_KEYS:
            _refuse("incomplete_original_metadata_binding")
        metadata = {key: _text(value, empty=key in {"state_schema_revision", "control_schema_profile_id", "control_schema_fingerprint"}, maximum=4096) for key, value in self.expected_metadata.items()}
        if metadata["interface"] != EXECUTION_INTERFACE or metadata["schema"] != EXECUTION_SCHEMA:
            _refuse("unsupported_execution_schema")
        if metadata["authority_mode"] not in {"embedded", "embedded_exclusive", "quack"} or metadata["logical_owner_session_id"] != self.owner_session_id:
            _refuse("original_owner_binding_mismatch")
        directory = _path(metadata["execution_state_dir"])
        prefix = metadata["execution_state_prefix"]
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", prefix) is None:
            _refuse("invalid_lane_binding")
        if re.fullmatch(r"[1-9][0-9]{0,4}", metadata["task_shard_count"]) is None or re.fullmatch(r"0|[1-9][0-9]{0,4}", metadata["task_shard_index"]) is None:
            _refuse("invalid_lane_binding")
        count, index = int(metadata["task_shard_count"]), int(metadata["task_shard_index"])
        if not 2 <= count <= 65535 or not 0 <= index < count or metadata["strict_task_sharding"] != "true":
            _refuse("unsupported_or_invalid_lane_binding")
        if path != directory / f"{prefix}.execution.shard-{index:04d}-of-{count:04d}.duckdb":
            _refuse("wrong_execution_lane_path")
        object.__setattr__(self, "expected_metadata", MappingProxyType(metadata))
        if not isinstance(self.source_bindings, Mapping) or not 1 <= len(self.source_bindings) <= 64:
            _refuse("missing_source_binding")
        sources: dict[str, str] = {}
        for raw_path, digest in self.source_bindings.items():
            source = _path(raw_path)
            if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None or source.suffix != ".py":
                _refuse("invalid_source_binding")
            sources[str(source)] = digest
        if sum(Path(source).name == "implementation_daemon.py" for source in sources) != 1:
            _refuse("missing_native_producer_source_binding")
        object.__setattr__(self, "source_bindings", MappingProxyType(sources))

    @property
    def binding_cid(self) -> str:
        return "sha256:" + hashlib.sha256(_json_bytes(self.to_dict())).hexdigest()

    @property
    def writer_lock_path(self) -> Path:
        path = Path(self.execution_path)
        return path.with_name(f".{path.name}.writer.lock")

    @property
    def original_producer_source(self) -> dict[str, str]:
        path = next(path for path in self.source_bindings if Path(path).name == "implementation_daemon.py")
        return {"path": path, "sha256": self.source_bindings[path]}

    def to_dict(self) -> dict[str, Any]:
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        for name in ("execution_file_identity", "writer_lock_identity", "expected_metadata", "source_bindings"):
            result[name] = dict(result[name])
        return {"schema": self.SCHEMA, **result}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NativeExecutionStatusBinding:
        if not isinstance(value, Mapping) or set(value) != {"schema", *(field.name for field in fields(cls))} or value.get("schema") != cls.SCHEMA:
            _refuse("invalid_status_binding_shape")
        return cls(**{key: value[key] for key in value if key != "schema"})

    def validate_control_task(self, task: Mapping[str, Any]) -> None:
        """Pure exact join for a real owner's task projection, before and after.

        The owner must independently bind its own store ID/schema. This method
        checks the already-decoded control receipt; it never treats an arbitrary
        mapping or a successful join as an authenticated capability.
        """
        if not isinstance(task, Mapping) or task.get("task_cid") != self.task_cid or type(task.get("revision")) is not int or task["revision"] != self.control_revision or task.get("status") != self.expected_control_status:
            _refuse("control_cursor_mismatch")
        body = task.get("body")
        if not isinstance(body, Mapping):
            _refuse("control_body_missing")
        receipt = body.get("completion_receipt")
        if not isinstance(receipt, Mapping) or receipt.get("operation") != "database_claim":
            _refuse("control_claim_receipt_missing")
        # The genuine database_claim receipt does not duplicate task_cid: the
        # primary task row binds it. Refuse a contradictory optional duplicate.
        if "task_cid" in receipt and receipt["task_cid"] != self.task_cid:
            _refuse("control_claim_join_mismatch")
        for name in ("attempt_id", "attempt_number", "claim_id", "lease_id", "owner_session_id", "fencing_token", "fence_epoch"):
            expected = getattr(self, name)
            if type(receipt.get(name)) is not type(expected) or receipt[name] != expected:
                _refuse("control_claim_join_mismatch")


class NativeExecutionStatusOwner:
    """Own a sealed readonly snapshot for one exact existing native attempt.

    No public connector, SQL, mutator, worker or provider callback is accepted.
    The genuine typed/Quack owner must hold source/resource custody and create
    an independently authenticated status scope before exposing observations.
    """

    def __init__(self, binding: NativeExecutionStatusBinding, *, deadline_monotonic: float) -> None:
        if type(binding) is not NativeExecutionStatusBinding:
            _refuse("invalid_native_status_binding")
        now = time.monotonic()
        if type(deadline_monotonic) not in (float, int) or not math.isfinite(deadline_monotonic) or not now < deadline_monotonic <= now + MAX_LIFETIME_SECONDS:
            _refuse("invalid_status_deadline")
        self._binding = binding
        self._deadline_monotonic = float(deadline_monotonic)
        self._lock = threading.RLock()
        self._connection: Any = None
        self._lock_fd: int | None = None
        self._database_fd: int | None = None
        self._opened = False
        self._closed = False
        self._cancelled = threading.Event()
        self._timer: threading.Timer | None = None

    @property
    def binding(self) -> NativeExecutionStatusBinding:
        return self._binding

    @property
    def deadline_monotonic(self) -> float:
        return self._deadline_monotonic

    def _check(self) -> None:
        if self._closed:
            _refuse("status_owner_closed")
        if self._cancelled.is_set():
            _refuse("status_owner_cancelled")
        if time.monotonic() >= self.deadline_monotonic:
            _refuse("status_deadline_expired")

    def _verify_files(self) -> None:
        self._check()
        try:
            if str(Path(__file__).absolute()) not in self.binding.source_bindings:
                _refuse("loaded_reader_source_not_bound")
            actual = _stat_identity(Path(self.binding.execution_path))
            lock = _stat_identity(self.binding.writer_lock_path)
            if actual != dict(self.binding.execution_file_identity) or lock != dict(self.binding.writer_lock_identity):
                _refuse("original_file_identity_drift")
            for source, expected in self.binding.source_bindings.items():
                path = Path(source)
                before = _stat_identity(path)
                if before["size"] > MAX_SOURCE_BYTES:
                    _refuse("source_bound_exceeded")
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
                try:
                    if _opened_identity(fd) != before:
                        _refuse("source_identity_drift")
                    digest = hashlib.sha256()
                    consumed = 0
                    while block := os.read(fd, 65_536):
                        self._check()
                        consumed += len(block)
                        if consumed > MAX_SOURCE_BYTES:
                            _refuse("source_bound_exceeded")
                        digest.update(block)
                    if _stat_identity(path) != before or digest.hexdigest() != expected:
                        _refuse("source_binding_drift")
                finally:
                    os.close(fd)
            # Never invoke writable recovery or remove a WAL to make a read pass.
            wal = Path(self.binding.execution_path + ".wal")
            if os.path.lexists(wal):
                _refuse("execution_wal_requires_native_review")
        except NativeExecutionStatusError:
            raise
        except OSError:
            _refuse("original_file_unavailable")

    def _rows(self, sql: str, values: list[Any] | None = None, *, maximum: int) -> list[tuple[Any, ...]]:
        self._check()
        try:
            rows = self._connection.execute(sql, values or []).fetchmany(maximum + 1)
        except Exception:
            self._check()
            _refuse("native_read_failed")
        self._check()
        if len(rows) > maximum or any(type(row) is not tuple for row in rows):
            _refuse("native_result_bound_exceeded")
        return rows

    def _verify_schema_metadata(self) -> dict[str, str]:
        columns = self._rows("""
            SELECT CASE WHEN octet_length(encode(table_name))<=128 THEN table_name ELSE NULL END,
              CASE WHEN octet_length(encode(column_name))<=128 THEN column_name ELSE NULL END,
              CASE WHEN octet_length(encode(data_type))<=32 THEN data_type ELSE NULL END,
              CASE WHEN octet_length(encode(is_nullable))<=3 THEN is_nullable ELSE NULL END
            FROM information_schema.columns
            WHERE table_catalog = current_database() AND table_schema = 'main'
            ORDER BY table_name, ordinal_position LIMIT 129
        """, maximum=128)
        actual: dict[str, list[tuple[str, str]]] = {}
        for table, column, kind, nullable in columns:
            if table in _COLUMNS:
                actual.setdefault(table, []).append((column, kind))
                if nullable != ("YES" if (table, column) == ("database_task_attempts", "finished_at_ms") else "NO"):
                    _refuse("execution_schema_nullable_mismatch")
        if {table: tuple(actual.get(table, ())) for table in _COLUMNS} != _COLUMNS:
            _refuse("execution_schema_mismatch")
        constraints = self._rows("""
            SELECT CASE WHEN octet_length(encode(t.table_name))<=128 THEN t.table_name ELSE NULL END,
              CASE WHEN octet_length(encode(t.constraint_name))<=256 THEN t.constraint_name ELSE NULL END,
              CASE WHEN octet_length(encode(t.constraint_type))<=32 THEN t.constraint_type ELSE NULL END,
              CASE WHEN octet_length(encode(k.column_name))<=128 THEN k.column_name ELSE NULL END
            FROM information_schema.table_constraints t
            JOIN information_schema.key_column_usage k
              ON t.constraint_catalog=k.constraint_catalog
             AND t.constraint_schema=k.constraint_schema
             AND t.constraint_name=k.constraint_name
             AND t.table_name=k.table_name
            WHERE t.table_catalog=current_database() AND t.table_schema='main'
              AND t.constraint_type IN ('PRIMARY KEY','UNIQUE')
            ORDER BY t.table_name,t.constraint_name,k.ordinal_position LIMIT 65
        """, maximum=64)
        grouped: dict[tuple[str, str, str], list[str]] = {}
        for table, name, kind, column in constraints:
            if any(value is None for value in (table, name, kind, column)):
                _refuse("execution_schema_name_bound_exceeded")
            if table in _COLUMNS:
                grouped.setdefault((table, name, kind), []).append(column)
        actual_constraints = {table: {(kind, tuple(names)) for (candidate, _name, kind), names in grouped.items() if candidate == table} for table in _COLUMNS}
        if actual_constraints != _CONSTRAINTS:
            _refuse("execution_schema_constraint_mismatch")
        rows = self._rows("""
            SELECT CASE WHEN octet_length(encode(key))<=128 THEN key ELSE NULL END,
                   CASE WHEN octet_length(encode(value))<=4096 THEN value ELSE NULL END
            FROM main.daemon_execution_metadata ORDER BY key LIMIT 33
        """, maximum=MAX_METADATA_ROWS)
        metadata: dict[str, str] = {}
        for key, value in rows:
            if type(key) is not str or type(value) is not str or key in metadata or len(value.encode("utf-8")) > 4096:
                _refuse("invalid_native_metadata")
            metadata[key] = value
        if not _METADATA_KEYS <= set(metadata):
            _refuse("native_identity_incomplete")
        if any(metadata.get(key) != value for key, value in self.binding.expected_metadata.items()):
            _refuse("native_metadata_binding_mismatch")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,511}", metadata["process_instance_id"]) is None:
            _refuse("native_process_identity_invalid")
        # The real Quack producer deliberately records an empty fingerprint;
        # legacy non-operational launches may record an empty profile too.
        # The primary owner must verify its genuine schema independently.
        return metadata

    def open(self) -> NativeExecutionStatusOwner:
        with self._lock:
            self._check()
            if self._opened:
                return self
            try:
                self._verify_files()
                self._lock_fd = os.open(self.binding.writer_lock_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
                if _opened_identity(self._lock_fd) != dict(self.binding.writer_lock_identity):
                    _refuse("opened_original_file_identity_mismatch")
                try:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    _refuse("original_execution_writer_active")
                self._database_fd = os.open(self.binding.execution_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
                for fd, expected in ((self._lock_fd, self.binding.writer_lock_identity), (self._database_fd, self.binding.execution_file_identity)):
                    if _opened_identity(fd) != dict(expected):
                        _refuse("opened_original_file_identity_mismatch")
                self._verify_files()
                self._timer = threading.Timer(max(0.0, self.deadline_monotonic - time.monotonic()), self.cancel)
                self._timer.daemon = True
                self._timer.start()
                import duckdb
                from ..task_sources import duckdb_state
                policy_path = str(Path(duckdb_state.__file__).absolute())
                policy_code = getattr(duckdb_state.connect_duckdb_with_policy, "__code__", None)
                if policy_path not in self.binding.source_bindings or policy_code is None or str(Path(policy_code.co_filename).absolute()) != policy_path:
                    _refuse("loaded_backend_policy_source_not_bound")
                try:
                    self._connection = duckdb_state.connect_duckdb_with_policy(
                        duckdb, Path(self.binding.execution_path), read_only=True,
                        configuration={"threads": 1, "memory_limit": "256MB"},
                    )
                except Exception:
                    _refuse("native_read_only_open_failed")
                self._check()
                self._verify_schema_metadata()
                self._verify_files()
                self._opened = True
                return self
            except BaseException:
                self._close_unlocked()
                raise

    @staticmethod
    def _body(row: tuple[Any, ...]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        raw, length = row[-2:]
        if type(length) is not int or not 0 <= length <= MAX_JSON_BYTES or raw is None:
            _refuse("json_bound_exceeded")
        return row[:-2], _json_object(raw)

    def observe_attempt(self) -> dict[str, Any]:
        with self._lock:
            self._check()
            if not self._opened:
                _refuse("status_owner_not_open")
            try:
                self._verify_files()
                self._connection.execute("BEGIN TRANSACTION")
                try:
                    metadata = self._verify_schema_metadata()
                    rows = self._rows("""
                        SELECT CASE WHEN octet_length(encode(attempt_id))<=1024 THEN attempt_id ELSE NULL END,
                          CASE WHEN octet_length(encode(claim_id))<=1024 THEN claim_id ELSE NULL END,
                          CASE WHEN octet_length(encode(task_cid))<=1024 THEN task_cid ELSE NULL END,
                          CASE WHEN octet_length(encode(task_alias))<=1024 THEN task_alias ELSE NULL END,
                          attempt_number,
                          CASE WHEN octet_length(encode(owner_session_id))<=1024 THEN owner_session_id ELSE NULL END,
                          fencing_token,fence_epoch,
                          CASE WHEN octet_length(encode(lease_id))<=1024 THEN lease_id ELSE NULL END,
                          CASE WHEN octet_length(encode(committed_phase))<=32 THEN committed_phase ELSE NULL END,
                          CASE WHEN octet_length(encode(status))<=32 THEN status ELSE NULL END,
                          started_at_ms,finished_at_ms,revision,
                          CASE WHEN octet_length(encode(body_json))<=16384 THEN body_json ELSE NULL END,
                          octet_length(encode(body_json))
                        FROM main.database_task_attempts WHERE attempt_id=? LIMIT 2
                    """, [self.binding.attempt_id], maximum=1)
                    attempt = None
                    if rows:
                        values, body = self._body(rows[0])
                        attempt = dict(zip(_ATTEMPT_COLUMNS[:-1], values))
                        attempt["body"] = body
                        _text(attempt["task_alias"], empty=True, maximum=1024)
                        for name in ("attempt_id", "claim_id", "task_cid", "attempt_number", "owner_session_id", "fencing_token", "fence_epoch", "lease_id"):
                            expected = getattr(self.binding, name)
                            if type(attempt.get(name)) is not type(expected) or attempt[name] != expected:
                                _refuse("native_attempt_join_mismatch")
                        if attempt["committed_phase"] not in _PHASES or attempt["status"] not in {"running", "succeeded", "failed", "blocked"}:
                            _refuse("native_attempt_cursor_invalid")
                        for name in ("revision", "started_at_ms"):
                            _integer(attempt[name])
                        if attempt["finished_at_ms"] is not None:
                            _integer(attempt["finished_at_ms"])
                    phase_rows = self._rows("""
                        SELECT CASE WHEN octet_length(encode(phase))<=32 THEN phase ELSE NULL END,
                          committed_at_ms,fencing_token,fence_epoch,revision,
                          CASE WHEN octet_length(encode(body_json))<=16384 THEN body_json ELSE NULL END,
                          octet_length(encode(body_json))
                        FROM main.attempt_phases WHERE attempt_id=?
                        ORDER BY committed_at_ms,phase LIMIT 9
                    """, [self.binding.attempt_id], maximum=MAX_PHASES)
                    phases = []
                    for row in phase_rows:
                        values, body = self._body(row)
                        phase = dict(zip(("phase", "committed_at_ms", "fencing_token", "fence_epoch", "revision"), values))
                        phase["body"] = body
                        if phase["phase"] not in _PHASES or any(type(phase[key]) is not int or phase[key] != getattr(self.binding, key) for key in ("fencing_token", "fence_epoch")):
                            _refuse("native_phase_join_mismatch")
                        _integer(phase["revision"])
                        _integer(phase["committed_at_ms"])
                        if attempt is None or phase["revision"] > attempt["revision"]:
                            _refuse("native_phase_cursor_inconsistent")
                        phases.append(phase)
                    if attempt is not None and (
                        not phases or len({phase["revision"] for phase in phases}) != len(phases)
                        or not any(phase["phase"] == "claimed" and phase["revision"] == 1 for phase in phases)
                        or not any(phase["phase"] == attempt["committed_phase"] and phase["revision"] == attempt["revision"] for phase in phases)
                    ):
                        _refuse("native_phase_cursor_inconsistent")
                    provider = self._records("provider")
                    effect = self._records("effect")
                    if attempt is None and (provider or effect):
                        _refuse("native_orphan_records")
                    result = {
                        "schema": OBSERVATION_SCHEMA, "binding_cid": self.binding.binding_cid,
                        "observation_only": True, "backend_read_only": True,
                        "native_store_uuid": None,
                        "native_store_identity_kind": "existing_file_and_native_execution_metadata",
                        "physical_identity_provenance": "current_server_selected_identity_not_historical_inode_attestation",
                        "preknown_original_process_instance_id": "process_instance_id" in self.binding.expected_metadata,
                        "observed_process_instance_id": metadata["process_instance_id"],
                        "original_process_impersonated": False,
                        "source_provenance": "server_selected_source_pins_not_historical_execution_attestation",
                        "original_producer_source": self.binding.original_producer_source,
                        "observer_source": {"path": str(Path(__file__).absolute()), "sha256": self.binding.source_bindings[str(Path(__file__).absolute())]},
                        "native_control_schema_profile_recorded": bool(metadata["control_schema_profile_id"]),
                        "native_control_schema_fingerprint_recorded": bool(metadata["control_schema_fingerprint"]),
                        "execution_file_identity": dict(self.binding.execution_file_identity),
                        "native_metadata": metadata, "attempt": attempt,
                        "attempt_observation": "present" if attempt is not None else "missing_record",
                        "phase_history": phases, "phase_history_complete": True,
                        "provider_records": provider, "effect_records": effect,
                        "provider_records_complete": True, "effect_records_complete": True,
                        "provider_effect_state": "unknown",
                        "native_callback_authority_observed": False,
                        "coordination_authority_observed": False,
                        "control_join_verified_by_component": False,
                        "completion_authority": False, "automatic_retry_admitted": False,
                        "semantic_admission": False,
                    }
                    if len(_json_bytes(result)) > MAX_OBSERVATION_BYTES:
                        _refuse("observation_bound_exceeded")
                except BaseException:
                    try:
                        self._connection.execute("ROLLBACK")
                    except Exception:
                        pass  # Preserve the original failure; close follows.
                    raise
                else:
                    try:
                        self._connection.execute("ROLLBACK")
                    except Exception:
                        _refuse("native_snapshot_release_failed")
                self._verify_files()
                return copy.deepcopy(result)
            except BaseException:
                self._close_unlocked()
                raise

    def _records(self, kind: str) -> list[dict[str, Any]]:
        # Closed internal choice, never a table/idempotency selector from RPC.
        if kind == "provider":
            rows = self._rows("""
                SELECT CASE WHEN octet_length(encode(invocation_id))<=1024 THEN invocation_id ELSE NULL END,
                  CASE WHEN octet_length(encode(attempt_id))<=1024 THEN attempt_id ELSE NULL END,
                  CASE WHEN octet_length(encode(task_cid))<=1024 THEN task_cid ELSE NULL END,
                  CASE WHEN octet_length(encode(idempotency_key))<=1024 THEN idempotency_key ELSE NULL END,
                  CASE WHEN octet_length(encode(owner_session_id))<=1024 THEN owner_session_id ELSE NULL END,
                  recorded_at_ms,
                  CASE WHEN octet_length(encode(result_json))<=16384 THEN result_json ELSE NULL END,
                  octet_length(encode(result_json))
                FROM main.provider_invocations WHERE attempt_id=? ORDER BY idempotency_key LIMIT 9
            """, [self.binding.attempt_id], maximum=MAX_IDEMPOTENCY_RECORDS)
            keys = ("invocation_id", "attempt_id", "task_cid", "idempotency_key", "owner_session_id", "recorded_at_ms")
        elif kind == "effect":
            rows = self._rows("""
                SELECT CASE WHEN octet_length(encode(effect_id))<=1024 THEN effect_id ELSE NULL END,
                  CASE WHEN octet_length(encode(attempt_id))<=1024 THEN attempt_id ELSE NULL END,
                  CASE WHEN octet_length(encode(task_cid))<=1024 THEN task_cid ELSE NULL END,
                  CASE WHEN octet_length(encode(effect_key))<=1024 THEN effect_key ELSE NULL END,
                  CASE WHEN octet_length(encode(idempotency_key))<=1024 THEN idempotency_key ELSE NULL END,
                  CASE WHEN octet_length(encode(owner_session_id))<=1024 THEN owner_session_id ELSE NULL END,
                  recorded_at_ms,
                  CASE WHEN octet_length(encode(result_json))<=16384 THEN result_json ELSE NULL END,
                  octet_length(encode(result_json))
                FROM main.effect_claims WHERE attempt_id=? ORDER BY idempotency_key LIMIT 9
            """, [self.binding.attempt_id], maximum=MAX_IDEMPOTENCY_RECORDS)
            keys = ("effect_id", "attempt_id", "task_cid", "effect_key", "idempotency_key", "owner_session_id", "recorded_at_ms")
        else:
            _refuse("invalid_internal_record_kind")
        records = []
        for row in rows:
            values, body = self._body(row)
            record = dict(zip(keys, values))
            record["result"] = body
            for name in ("attempt_id", "task_cid", "owner_session_id"):
                if record[name] != getattr(self.binding, name):
                    _refuse("native_idempotency_join_mismatch")
            _integer(record["recorded_at_ms"])
            for name in keys:
                if name != "recorded_at_ms":
                    _text(record[name], maximum=1024)
            records.append(record)
        return records

    def _close_unlocked(self) -> None:
        self._closed = True
        self._opened = False
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()
        connection = self._connection
        if connection is not None:
            try:
                connection.close()
            except Exception:
                # Do not release the original writer guard around an unknown
                # live native handle. A real owner STOP must report failure.
                _refuse("native_connection_close_unverified")
            self._connection = None
        for name in ("_database_fd", "_lock_fd"):
            fd = getattr(self, name)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)

    def close(self) -> None:
        with self._lock:
            self._close_unlocked()

    def cancel(self) -> None:
        self._cancelled.set()
        connection = self._connection
        if connection is not None:
            try:
                connection.interrupt()
            except Exception:
                pass
        self.close()

    def __enter__(self) -> NativeExecutionStatusOwner:
        return self.open()

    def __exit__(self, *_exc: Any) -> None:
        self.close()
