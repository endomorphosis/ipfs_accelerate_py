"""Authenticated, bounded envelopes for Quack owner-side mutations.

The Quack beta transport can read and insert attached rows, but some builds
cannot update or delete base tables.  Those operations are handed to the
exclusive state owner through a local, mode-0700 inbox.  This module keeps the
handoff typed and authenticated without ever writing the raw Quack token.

DOEP-042 extends the same owner-mutation carrier with claim, lease, fence, and
idempotency admission.  It consumes ``CanonicalTaskStateMachine@1`` and
``StateTransaction@1`` rather than creating a second writer, lease coordinator,
or idempotency store.  A worker or model assertion cannot authorize a mutation.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import hmac
import json
import math
import os
import re
import stat as stat_module
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .control_plane_contracts import (
    CANONICAL_TASK_STATE_MACHINE_INTERFACE,
    CommandOutcome,
    ControlPlaneContractError,
    TaskState,
    TaskStateSnapshot,
)
from .control_plane_transactions import (
    STATE_TRANSACTION_INTERFACE,
    FenceMismatchError,
    IdempotencyConflictError,
    TransactionError,
    assert_task_cas_transition,
)

QUACK_OWNER_MUTATION_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-owner-mutation-request@1"
)
QUACK_OWNER_MUTATION_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-owner-mutation-result@1"
)
QUACK_OWNER_MUTATION_INTERFACE: Final = "QuackOwnerMutation@1"
CLAIM_LEASE_FENCE_IDEMPOTENCY_BINDING: Final = "ClaimLeaseFenceIdempotency@1"
CLAIM_LEASE_FENCE_IDEMPOTENCY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/claim-lease-fence-idempotency@1"
)
LIVE_OWNER_CLAIM_LEASE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/live-owner-claim-lease@1"
)
OWNER_MUTATION_AUTHORITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/owner-mutation-authority@1"
)
ACCEPTED_LEASE_STATE: Final = "accepted"
MIN_LIVE_FENCE_EPOCH: Final[int] = 1
MIN_LIVE_FENCING_TOKEN: Final[int] = 1

MAX_MUTATION_SQL_BYTES: Final[int] = 65_536
MAX_MUTATION_PARAMETERS_BYTES: Final[int] = 262_144
MAX_MUTATION_REQUEST_BYTES: Final[int] = 393_216
MAX_MUTATION_RESULT_BYTES: Final[int] = 262_144
MAX_MUTATION_RESULT_ROWS: Final[int] = 128
MAX_MUTATION_REQUEST_AGE_MS: Final[int] = 60_000
MAX_MUTATION_FUTURE_SKEW_MS: Final[int] = 5_000

_REQUEST_ID_RE: Final = re.compile(r"^[0-9a-f]{32}$")
_SIGNATURE_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_RE: Final = re.compile(r"^[A-Za-z0-9_-]{8,}$")
_DIGEST_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_ADMITTED_PREFIXES: Final = (
    "UPDATE ",
    "DELETE ",
    "MERGE ",
    "INSERT OR REPLACE ",
    "INSERT OR IGNORE ",
)
_REQUEST_FIELDS: Final = frozenset(
    {
        "schema",
        "request_id",
        "store_id",
        "generation",
        "issued_at_ms",
        "sql",
        "parameters",
        "signature",
    }
)
_RESULT_FIELDS: Final = frozenset(
    {
        "schema",
        "request_id",
        "store_id",
        "generation",
        "completed_at_ms",
        "ok",
        "rowcount",
        "columns",
        "rows",
        "error_code",
        "error",
        "signature",
    }
)

_AT_FDCWD: Final[int] = -100
_RENAME_NOREPLACE: Final[int] = 1
try:
    _LIBC = ctypes.CDLL(None, use_errno=True)
    _RENAMEAT2 = getattr(_LIBC, "renameat2", None)
except OSError:  # pragma: no cover - fail-closed platform boundary
    _RENAMEAT2 = None
if _RENAMEAT2 is not None:
    _RENAMEAT2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    _RENAMEAT2.restype = ctypes.c_int


class QuackOwnerMutationEnvelopeError(ValueError):
    """A mutation request/result failed its closed envelope contract."""

    def __init__(self, message: str, *, code: str = "malformed_envelope") -> None:
        super().__init__(message)
        self.code = str(code)


def _publish_without_replace(source: Path, target: Path) -> None:
    """Atomically publish one complete file while preserving collision denial."""

    if _RENAMEAT2 is None:
        raise QuackOwnerMutationEnvelopeError(
            "atomic no-replace publication is unavailable",
            code="atomic_publication_unavailable",
        )
    ctypes.set_errno(0)
    result = _RENAMEAT2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(target),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno() or errno.EIO
        raise OSError(error_number, os.strerror(error_number), str(target))


def _publish_without_replace_at(
    directory_fd: int,
    source_name: str,
    target_name: str,
) -> None:
    """Publish within one already-admitted directory without replacement."""

    if _RENAMEAT2 is None:
        raise QuackOwnerMutationEnvelopeError(
            "atomic no-replace publication is unavailable",
            code="atomic_publication_unavailable",
        )
    ctypes.set_errno(0)
    result = _RENAMEAT2(
        int(directory_fd),
        os.fsencode(source_name),
        int(directory_fd),
        os.fsencode(target_name),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno() or errno.EIO
        raise OSError(error_number, os.strerror(error_number), target_name)


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
    )


def open_mutation_inbox_directory(path: Path) -> int:
    """Create and pin one absolute inbox through no-follow directory handles.

    The returned descriptor is the authority for subsequent envelope I/O.
    Callers must close it after the complete worker request or owner pump
    cycle, rather than reopening ``path`` after admission.
    """

    target = Path(path).expanduser()
    if not target.is_absolute() or not target.name or "\x00" in os.fspath(target):
        raise QuackOwnerMutationEnvelopeError(
            "mutation inbox path must be an absolute directory",
            code="unsafe_inbox",
        )
    if any(component == ".." for component in target.parts):
        raise QuackOwnerMutationEnvelopeError(
            "mutation inbox path contains parent traversal",
            code="unsafe_inbox",
        )
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise QuackOwnerMutationEnvelopeError(
            "mutation inbox requires no-follow directory support",
            code="unsafe_inbox",
        )
    flags = os.O_RDONLY | nofollow | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = -1
    try:
        descriptor = os.open(target.anchor, flags)
        for component in target.parts[1:]:
            child = -1
            try:
                try:
                    child = os.open(component, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    except FileExistsError:
                        # A concurrent creator is acceptable only when the
                        # exact entry can now be opened no-follow.
                        pass
                    child = os.open(component, flags, dir_fd=descriptor)
                opened = os.fstat(child)
                named = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat_module.S_ISDIR(opened.st_mode)
                    or not stat_module.S_ISDIR(named.st_mode)
                    or _directory_identity(opened) != _directory_identity(named)
                ):
                    raise QuackOwnerMutationEnvelopeError(
                        "mutation inbox path component is unsafe",
                        code="unsafe_inbox",
                    )
            except BaseException:
                if child >= 0:
                    os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        os.fchmod(descriptor, 0o700)
        admitted = os.fstat(descriptor)
        if (
            not stat_module.S_ISDIR(admitted.st_mode)
            or admitted.st_uid != os.geteuid()
            or stat_module.S_IMODE(admitted.st_mode) != 0o700
        ):
            raise QuackOwnerMutationEnvelopeError(
                "mutation inbox identity or ownership is unsafe",
                code="unsafe_inbox",
            )
        pinned = descriptor
        descriptor = -1
        return pinned
    except QuackOwnerMutationEnvelopeError:
        raise
    except OSError as exc:
        raise QuackOwnerMutationEnvelopeError(
            "mutation inbox is not a safe owner directory",
            code="unsafe_inbox",
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _entry_name(value: str | os.PathLike[str]) -> str:
    name = os.fspath(value)
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "\x00" in name
        or os.path.basename(name) != name
    ):
        raise QuackOwnerMutationEnvelopeError(
            "mutation envelope filename is unsafe",
            code="malformed_filename",
        )
    return name


def mutation_envelope_exists_at(
    directory_fd: int,
    name: str | os.PathLike[str],
) -> bool:
    """Return whether an entry exists in one pinned inbox, without following it."""

    filename = _entry_name(name)
    try:
        os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def unlink_mutation_envelope_at(
    directory_fd: int,
    name: str | os.PathLike[str],
    *,
    missing_ok: bool = False,
) -> None:
    """Remove one entry relative to the pinned inbox descriptor."""

    filename = _entry_name(name)
    try:
        os.unlink(filename, dir_fd=directory_fd)
    except FileNotFoundError:
        if not missing_ok:
            raise


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise QuackOwnerMutationEnvelopeError("mutation envelope is not canonical JSON") from exc
    return encoded


def _token_bytes(token: str) -> bytes:
    value = str(token or "").strip()
    if not _TOKEN_RE.fullmatch(value):
        raise QuackOwnerMutationEnvelopeError(
            "Quack mutation authentication token is unavailable",
            code="authentication_unavailable",
        )
    return value.encode("ascii")


def _signature(payload: Mapping[str, Any], token: str) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    return hmac.new(
        _token_bytes(token),
        _canonical_bytes(unsigned),
        hashlib.sha256,
    ).hexdigest()


def _strict_fields(payload: Mapping[str, Any], expected: frozenset[str]) -> None:
    actual = frozenset(str(key) for key in payload)
    if actual != expected:
        unknown = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise QuackOwnerMutationEnvelopeError(
            f"mutation envelope fields differ (unknown={unknown}, missing={missing})"
        )


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise QuackOwnerMutationEnvelopeError(f"{name} must be a positive integer")
    return int(value)


def _request_id(value: Any) -> str:
    text = str(value or "")
    if not _REQUEST_ID_RE.fullmatch(text):
        raise QuackOwnerMutationEnvelopeError("request_id must be 32 lowercase hex digits")
    return text


def _identity(value: Any, *, name: str) -> str:
    text = str(value or "").strip()
    if not text or len(text.encode("utf-8")) > 256 or "\x00" in text:
        raise QuackOwnerMutationEnvelopeError(f"{name} is missing or unbounded")
    return text


def admit_mutation_sql(sql: Any) -> str:
    """Return one bounded DML statement or reject it before execution."""

    if not isinstance(sql, str):
        raise QuackOwnerMutationEnvelopeError("sql must be a string")
    statement = sql.strip()
    if not statement or len(statement.encode("utf-8")) > MAX_MUTATION_SQL_BYTES:
        raise QuackOwnerMutationEnvelopeError("sql is empty or exceeds its bound")
    if "\x00" in statement:
        raise QuackOwnerMutationEnvelopeError("sql contains a NUL byte")
    without_terminal = statement[:-1].rstrip() if statement.endswith(";") else statement
    if ";" in without_terminal:
        raise QuackOwnerMutationEnvelopeError(
            "multiple SQL statements are forbidden", code="sql_not_admitted"
        )
    normalized = " ".join(without_terminal.upper().split())
    if not normalized.startswith(_ADMITTED_PREFIXES):
        raise QuackOwnerMutationEnvelopeError(
            "SQL operation is not admitted by the owner mutation policy",
            code="sql_not_admitted",
        )
    return statement


def _parameters(value: Any) -> Any:
    if value is not None and not isinstance(value, (list, dict)):
        raise QuackOwnerMutationEnvelopeError("parameters must be null, an array, or an object")
    encoded = _canonical_bytes({"parameters": value})
    if len(encoded) > MAX_MUTATION_PARAMETERS_BYTES:
        raise QuackOwnerMutationEnvelopeError("parameters exceed their byte bound")
    # Round-trip to detach mutable caller objects and reject non-JSON values.
    return json.loads(encoded)["parameters"]


def build_mutation_request(
    *,
    request_id: str,
    store_id: str,
    generation: int,
    sql: str,
    parameters: Any,
    token: str,
    issued_at_ms: int | None = None,
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "schema": QUACK_OWNER_MUTATION_REQUEST_SCHEMA,
        "request_id": _request_id(request_id),
        "store_id": _identity(store_id, name="store_id"),
        "generation": _positive_int(generation, name="generation"),
        "issued_at_ms": _positive_int(
            int(time.time() * 1000) if issued_at_ms is None else issued_at_ms,
            name="issued_at_ms",
        ),
        "sql": admit_mutation_sql(sql),
        "parameters": _parameters(parameters),
    }
    payload["signature"] = _signature(payload, token)
    return MappingProxyType(payload)


def parse_mutation_request(
    payload: Mapping[str, Any],
    *,
    token: str,
    expected_request_id: str,
    expected_store_id: str,
    expected_generation: int,
    now_ms: int | None = None,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise QuackOwnerMutationEnvelopeError("mutation request must be an object")
    _strict_fields(payload, _REQUEST_FIELDS)
    if payload.get("schema") != QUACK_OWNER_MUTATION_REQUEST_SCHEMA:
        raise QuackOwnerMutationEnvelopeError("unknown mutation request schema")
    request_id = _request_id(payload.get("request_id"))
    signature = str(payload.get("signature") or "")
    if not _SIGNATURE_RE.fullmatch(signature) or not hmac.compare_digest(
        signature, _signature(payload, token)
    ):
        raise QuackOwnerMutationEnvelopeError(
            "mutation request authentication failed",
            code="authentication_failed",
        )
    if request_id != _request_id(expected_request_id):
        raise QuackOwnerMutationEnvelopeError(
            "request identity does not match its filename", code="identity_mismatch"
        )
    store_id = _identity(payload.get("store_id"), name="store_id")
    generation = _positive_int(payload.get("generation"), name="generation")
    if store_id != _identity(expected_store_id, name="expected_store_id"):
        raise QuackOwnerMutationEnvelopeError(
            "mutation request targets another store", code="identity_mismatch"
        )
    if generation != _positive_int(expected_generation, name="expected_generation"):
        raise QuackOwnerMutationEnvelopeError(
            "mutation request targets another generation", code="identity_mismatch"
        )
    issued_at = _positive_int(payload.get("issued_at_ms"), name="issued_at_ms")
    observed = int(time.time() * 1000) if now_ms is None else int(now_ms)
    if issued_at < observed - MAX_MUTATION_REQUEST_AGE_MS:
        raise QuackOwnerMutationEnvelopeError("mutation request is stale", code="stale_request")
    if issued_at > observed + MAX_MUTATION_FUTURE_SKEW_MS:
        raise QuackOwnerMutationEnvelopeError(
            "mutation request is from the future", code="stale_request"
        )
    normalized = {
        "schema": QUACK_OWNER_MUTATION_REQUEST_SCHEMA,
        "request_id": request_id,
        "store_id": store_id,
        "generation": generation,
        "issued_at_ms": issued_at,
        "sql": admit_mutation_sql(payload.get("sql")),
        "parameters": _parameters(payload.get("parameters")),
        "signature": signature,
    }
    return MappingProxyType(normalized)


def _result_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise QuackOwnerMutationEnvelopeError(
        "mutation result contains a non-canonical scalar",
        code="result_not_serializable",
    )


def build_mutation_result(
    *,
    request_id: str,
    store_id: str,
    generation: int,
    ok: bool,
    token: str,
    rowcount: int = -1,
    columns: Sequence[str] = (),
    rows: Sequence[Sequence[Any]] = (),
    error_code: str = "",
    error: str = "",
    completed_at_ms: int | None = None,
) -> Mapping[str, Any]:
    if type(ok) is not bool:
        raise QuackOwnerMutationEnvelopeError("ok must be a boolean")
    names = [str(item) for item in columns]
    if len(names) > 128 or any(not item or len(item) > 256 for item in names):
        raise QuackOwnerMutationEnvelopeError("result columns exceed their bound")
    if len(rows) > MAX_MUTATION_RESULT_ROWS:
        raise QuackOwnerMutationEnvelopeError("result rows exceed their bound")
    normalized_rows = [[_result_scalar(value) for value in row] for row in rows]
    if any(len(row) != len(names) for row in normalized_rows):
        raise QuackOwnerMutationEnvelopeError("result row width does not match columns")
    payload: dict[str, Any] = {
        "schema": QUACK_OWNER_MUTATION_RESULT_SCHEMA,
        "request_id": _request_id(request_id),
        "store_id": _identity(store_id, name="store_id"),
        "generation": _positive_int(generation, name="generation"),
        "completed_at_ms": _positive_int(
            int(time.time() * 1000) if completed_at_ms is None else completed_at_ms,
            name="completed_at_ms",
        ),
        "ok": ok,
        "rowcount": int(rowcount),
        "columns": names,
        "rows": normalized_rows,
        "error_code": str(error_code or "")[:128],
        "error": str(error or "")[:1000],
    }
    if ok and (payload["error_code"] or payload["error"]):
        raise QuackOwnerMutationEnvelopeError("successful result cannot contain an error")
    if not ok and not payload["error_code"]:
        raise QuackOwnerMutationEnvelopeError("failed result requires error_code")
    if len(_canonical_bytes(payload)) > MAX_MUTATION_RESULT_BYTES:
        raise QuackOwnerMutationEnvelopeError("result exceeds its byte bound")
    payload["signature"] = _signature(payload, token)
    return MappingProxyType(payload)


def parse_mutation_result(
    payload: Mapping[str, Any],
    *,
    token: str,
    expected_request_id: str,
    expected_store_id: str,
    expected_generation: int,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise QuackOwnerMutationEnvelopeError("mutation result must be an object")
    _strict_fields(payload, _RESULT_FIELDS)
    if payload.get("schema") != QUACK_OWNER_MUTATION_RESULT_SCHEMA:
        raise QuackOwnerMutationEnvelopeError("unknown mutation result schema")
    signature = str(payload.get("signature") or "")
    if not _SIGNATURE_RE.fullmatch(signature) or not hmac.compare_digest(
        signature, _signature(payload, token)
    ):
        raise QuackOwnerMutationEnvelopeError(
            "mutation result authentication failed", code="authentication_failed"
        )
    request_id = _request_id(payload.get("request_id"))
    store_id = _identity(payload.get("store_id"), name="store_id")
    generation = _positive_int(payload.get("generation"), name="generation")
    if (
        request_id != _request_id(expected_request_id)
        or store_id != _identity(expected_store_id, name="expected_store_id")
        or generation != _positive_int(expected_generation, name="expected_generation")
    ):
        raise QuackOwnerMutationEnvelopeError(
            "mutation result identity does not match request",
            code="identity_mismatch",
        )
    # Rebuild through the constructor to apply every payload bound.
    normalized = build_mutation_result(
        request_id=request_id,
        store_id=store_id,
        generation=generation,
        ok=payload.get("ok"),
        token=token,
        rowcount=payload.get("rowcount", -1),
        columns=payload.get("columns") or (),
        rows=payload.get("rows") or (),
        error_code=payload.get("error_code") or "",
        error=payload.get("error") or "",
        completed_at_ms=payload.get("completed_at_ms"),
    )
    return MappingProxyType({**dict(normalized), "signature": signature})


def write_envelope_atomic_at(
    directory_fd: int,
    name: str | os.PathLike[str],
    payload: Mapping[str, Any],
    *,
    replace: bool,
) -> None:
    """Write one envelope relative to a pinned, admitted inbox descriptor."""

    target_name = _entry_name(name)
    data = _canonical_bytes(payload) + b"\n"
    temporary_name = f".{target_name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary_name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        stream = os.fdopen(descriptor, "wb")
        descriptor = -1
        with stream:
            stream.write(data)
            stream.flush()
            os.fchmod(stream.fileno(), 0o600)
            os.fsync(stream.fileno())
        if replace:
            os.replace(
                temporary_name,
                target_name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
        else:
            _publish_without_replace_at(
                directory_fd,
                temporary_name,
                target_name,
            )
        os.fsync(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _regular_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def read_envelope_at(
    directory_fd: int,
    name: str | os.PathLike[str],
    *,
    max_bytes: int = MAX_MUTATION_RESULT_BYTES,
) -> Mapping[str, Any]:
    """Read one stable envelope relative to a pinned inbox descriptor."""

    filename = _entry_name(name)
    descriptor = -1
    try:
        descriptor = os.open(
            filename,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        before = os.fstat(descriptor)
        if (
            not stat_module.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or stat_module.S_IMODE(before.st_mode) & 0o077
            or before.st_size < 2
            or before.st_size > max_bytes
        ):
            raise QuackOwnerMutationEnvelopeError("mutation envelope file is unbounded or unsafe")
        remaining = max_bytes + 1
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        after = os.fstat(descriptor)
        named = os.stat(
            filename,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            len(encoded) != before.st_size
            or _regular_identity(before) != _regular_identity(after)
            or _regular_identity(after) != _regular_identity(named)
        ):
            raise QuackOwnerMutationEnvelopeError("mutation envelope changed while being read")
        payload = json.loads(encoded.decode("utf-8"))
    except QuackOwnerMutationEnvelopeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QuackOwnerMutationEnvelopeError("mutation envelope JSON is malformed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(payload, Mapping):
        raise QuackOwnerMutationEnvelopeError("mutation envelope must contain an object")
    return MappingProxyType(dict(payload))


def write_envelope_atomic(path: Path, payload: Mapping[str, Any], *, replace: bool) -> None:
    """Write one mode-0600 canonical envelope without exposing partial JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        target.parent.chmod(0o700)
    except OSError:
        pass
    data = _canonical_bytes(payload) + b"\n"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, target)
        else:
            # ``RENAME_NOREPLACE`` makes the complete fsynced file visible in
            # one operation with link count one.  It also preserves collision
            # denial; no existing request can be overwritten or replayed.
            _publish_without_replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _natural_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise QuackOwnerMutationEnvelopeError(
            f"{name} must be an integer >= {minimum}",
            code="malformed_envelope",
        )
    return int(value)


def _digest(value: Any, *, name: str) -> str:
    text = str(value or "").strip()
    if not _DIGEST_RE.fullmatch(text):
        raise QuackOwnerMutationEnvelopeError(
            f"{name} must be a sha256 digest",
            code="malformed_envelope",
        )
    return text


def mutation_request_digest(*, sql: str, parameters: Any) -> str:
    """Return the exact mutation payload digest used for idempotent replay."""

    admitted = admit_mutation_sql(sql)
    bound = _parameters(parameters)
    return (
        "sha256:"
        + hashlib.sha256(
            _canonical_bytes({"sql": admitted, "parameters": bound})
        ).hexdigest()
    )


@dataclass(frozen=True)
class OwnerMutationAuthority:
    """Caller claim/lease/fence/idempotency binding for one owner mutation.

    This is not a second lease coordinator.  It is the exact set of values the
    exclusive owner must observe as live before executing envelope SQL.
    """

    SCHEMA: ClassVar[str] = OWNER_MUTATION_AUTHORITY_SCHEMA

    task_cid: str
    claim_id: str
    lease_id: str
    claimant_did: str
    fencing_token: int
    fence_epoch: int
    owner_session_id: str
    idempotency_key: str
    request_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_cid", _identity(self.task_cid, name="task_cid"))
        object.__setattr__(self, "claim_id", _identity(self.claim_id, name="claim_id"))
        object.__setattr__(self, "lease_id", _identity(self.lease_id, name="lease_id"))
        object.__setattr__(
            self, "claimant_did", _identity(self.claimant_did, name="claimant_did")
        )
        object.__setattr__(
            self,
            "fencing_token",
            _natural_int(
                self.fencing_token,
                name="fencing_token",
                minimum=MIN_LIVE_FENCING_TOKEN,
            ),
        )
        object.__setattr__(
            self,
            "fence_epoch",
            _natural_int(
                self.fence_epoch,
                name="fence_epoch",
                minimum=MIN_LIVE_FENCE_EPOCH,
            ),
        )
        object.__setattr__(
            self,
            "owner_session_id",
            _identity(self.owner_session_id, name="owner_session_id"),
        )
        object.__setattr__(
            self,
            "idempotency_key",
            _identity(self.idempotency_key, name="idempotency_key"),
        )
        object.__setattr__(
            self,
            "request_digest",
            _digest(self.request_digest, name="request_digest"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "task_cid": self.task_cid,
            "claim_id": self.claim_id,
            "lease_id": self.lease_id,
            "claimant_did": self.claimant_did,
            "fencing_token": self.fencing_token,
            "fence_epoch": self.fence_epoch,
            "owner_session_id": self.owner_session_id,
            "idempotency_key": self.idempotency_key,
            "request_digest": self.request_digest,
        }


@dataclass(frozen=True)
class LiveOwnerClaimLease:
    """Live exclusive-owner view of the canonical ``leases`` row.

    ``lease_id`` is the live ``claim_cid``, matching
    ``StateTransaction.assert_live_authorized_command_lease``.  Optional path
    claims must share the same task, fence, and owner session.
    """

    SCHEMA: ClassVar[str] = LIVE_OWNER_CLAIM_LEASE_SCHEMA

    task_cid: str
    claim_id: str
    lease_id: str
    claimant_did: str
    fencing_token: int
    fence_epoch: int
    expires_at_ms: int
    state: str
    owner_session_id: str
    revision: int
    path_claim_id: str = ""
    path_claim_state: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_cid", _identity(self.task_cid, name="task_cid"))
        object.__setattr__(self, "claim_id", _identity(self.claim_id, name="claim_id"))
        object.__setattr__(self, "lease_id", _identity(self.lease_id, name="lease_id"))
        object.__setattr__(
            self, "claimant_did", _identity(self.claimant_did, name="claimant_did")
        )
        object.__setattr__(
            self,
            "fencing_token",
            _natural_int(
                self.fencing_token,
                name="fencing_token",
                minimum=MIN_LIVE_FENCING_TOKEN,
            ),
        )
        object.__setattr__(
            self,
            "fence_epoch",
            _natural_int(
                self.fence_epoch,
                name="fence_epoch",
                minimum=MIN_LIVE_FENCE_EPOCH,
            ),
        )
        object.__setattr__(
            self,
            "expires_at_ms",
            _natural_int(self.expires_at_ms, name="expires_at_ms", minimum=1),
        )
        state = str(self.state or "").strip()
        if not state or len(state.encode("utf-8")) > 64:
            raise QuackOwnerMutationEnvelopeError(
                "live lease state is missing or unbounded",
                code="lease_mismatch",
            )
        object.__setattr__(self, "state", state)
        object.__setattr__(
            self,
            "owner_session_id",
            _identity(self.owner_session_id, name="owner_session_id"),
        )
        object.__setattr__(
            self,
            "revision",
            _natural_int(self.revision, name="revision", minimum=0),
        )
        path_claim_id = str(self.path_claim_id or "").strip()
        path_claim_state = str(self.path_claim_state or "").strip()
        if path_claim_id:
            path_claim_id = _identity(path_claim_id, name="path_claim_id")
        if path_claim_state and len(path_claim_state.encode("utf-8")) > 64:
            raise QuackOwnerMutationEnvelopeError(
                "path claim state is unbounded",
                code="claim_mismatch",
            )
        object.__setattr__(self, "path_claim_id", path_claim_id)
        object.__setattr__(self, "path_claim_state", path_claim_state)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "task_cid": self.task_cid,
            "claim_id": self.claim_id,
            "lease_id": self.lease_id,
            "claimant_did": self.claimant_did,
            "fencing_token": self.fencing_token,
            "fence_epoch": self.fence_epoch,
            "expires_at_ms": self.expires_at_ms,
            "state": self.state,
            "owner_session_id": self.owner_session_id,
            "revision": self.revision,
            "path_claim_id": self.path_claim_id,
            "path_claim_state": self.path_claim_state,
        }


@dataclass(frozen=True)
class OwnerMutationAdmission:
    """Typed owner-mutation admission or exact-digest idempotent replay."""

    SCHEMA: ClassVar[str] = CLAIM_LEASE_FENCE_IDEMPOTENCY_SCHEMA
    INTERFACE: ClassVar[str] = CLAIM_LEASE_FENCE_IDEMPOTENCY_BINDING

    outcome: CommandOutcome
    changed: bool
    requested: OwnerMutationAuthority
    live: LiveOwnerClaimLease
    replay_result: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        outcome = self.outcome
        if not isinstance(outcome, CommandOutcome):
            outcome = CommandOutcome(str(outcome))
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "changed", bool(self.changed))
        if not isinstance(self.requested, OwnerMutationAuthority):
            raise ControlPlaneContractError(
                "owner mutation admission requires OwnerMutationAuthority"
            )
        if not isinstance(self.live, LiveOwnerClaimLease):
            raise ControlPlaneContractError(
                "owner mutation admission requires LiveOwnerClaimLease"
            )
        replay = self.replay_result
        if replay is None:
            object.__setattr__(self, "replay_result", None)
        elif isinstance(replay, Mapping):
            object.__setattr__(self, "replay_result", MappingProxyType(dict(replay)))
        else:
            raise QuackOwnerMutationEnvelopeError(
                "idempotent replay result must be an object",
                code="idempotency_conflict",
            )
        if outcome is CommandOutcome.IDEMPOTENT_REPLAY:
            if self.changed or self.replay_result is None:
                raise QuackOwnerMutationEnvelopeError(
                    "idempotent replay requires an unchanged prior result",
                    code="idempotency_conflict",
                )
        elif outcome is CommandOutcome.ACCEPTED:
            if not self.changed or self.replay_result is not None:
                raise QuackOwnerMutationEnvelopeError(
                    "admitted mutation cannot carry a replay result",
                    code="malformed_envelope",
                )
        else:
            raise ControlPlaneContractError(
                "owner mutation admission outcome must be accepted or idempotent_replay"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "binding": self.INTERFACE,
            "carrier": QUACK_OWNER_MUTATION_INTERFACE,
            "consumes": {
                "task_state_machine": CANONICAL_TASK_STATE_MACHINE_INTERFACE,
                "state_transaction": STATE_TRANSACTION_INTERFACE,
            },
            "outcome": self.outcome.value,
            "changed": self.changed,
            "requested": self.requested.to_dict(),
            "live": self.live.to_dict(),
            "replay_result": None if self.replay_result is None else dict(self.replay_result),
            "worker_assertion_is_authority": False,
        }


def _existing_idempotency_digest(existing: Mapping[str, Any]) -> str:
    for name in ("request_digest", "result_digest"):
        raw = existing.get(name)
        if raw not in (None, ""):
            return _digest(raw, name=name)
    raise QuackOwnerMutationEnvelopeError(
        "idempotency record is missing a request digest",
        code="idempotency_conflict",
    )


def _existing_idempotency_result(existing: Mapping[str, Any]) -> Mapping[str, Any]:
    body = existing.get("result")
    if body is None:
        body = existing.get("body")
    if body is None:
        return MappingProxyType({})
    if not isinstance(body, Mapping):
        raise QuackOwnerMutationEnvelopeError(
            "idempotency record body is not an object",
            code="idempotency_conflict",
        )
    return MappingProxyType(dict(body))


def assert_live_claim_lease_fence(
    requested: OwnerMutationAuthority,
    live: LiveOwnerClaimLease,
    *,
    now_ms: int,
    worker_assertion: bool = False,
) -> LiveOwnerClaimLease:
    """Fail closed unless the caller holds the current claim, lease, and fence.

    Consumes the same live ``leases`` identity as
    ``StateTransaction.assert_live_authorized_command_lease``.  Worker or
    model assertions never authorize a stale or revoked binding.
    """

    if not isinstance(requested, OwnerMutationAuthority) or not isinstance(
        live, LiveOwnerClaimLease
    ):
        raise ControlPlaneContractError(
            "claim/lease/fence admission requires canonical owner bindings"
        )
    observed = _natural_int(now_ms, name="now_ms", minimum=1)
    # Diagnostic only; never a bypass of live claim, lease, or fence.
    _ = worker_assertion
    if requested.task_cid != live.task_cid:
        raise QuackOwnerMutationEnvelopeError(
            "mutation claim targets another task",
            code="claim_mismatch",
        )
    if requested.claim_id != live.claim_id and (
        not live.path_claim_id or requested.claim_id != live.path_claim_id
    ):
        raise QuackOwnerMutationEnvelopeError(
            "mutation claim identity does not match the live claim",
            code="claim_mismatch",
        )
    if live.path_claim_id and requested.claim_id == live.path_claim_id:
        if live.path_claim_state != ACCEPTED_LEASE_STATE:
            raise QuackOwnerMutationEnvelopeError(
                "mutation path claim is revoked",
                code="claim_mismatch",
            )
    if requested.lease_id != live.lease_id:
        raise QuackOwnerMutationEnvelopeError(
            "mutation lease identity does not match the live lease",
            code="lease_mismatch",
        )
    if requested.claimant_did != live.claimant_did:
        raise QuackOwnerMutationEnvelopeError(
            "mutation lease principal does not match the live claimant",
            code="lease_mismatch",
        )
    if requested.owner_session_id != live.owner_session_id:
        raise QuackOwnerMutationEnvelopeError(
            "mutation owner session does not match the live lease",
            code="lease_mismatch",
        )
    if live.state != ACCEPTED_LEASE_STATE:
        raise QuackOwnerMutationEnvelopeError(
            "authorized owner mutation lease is revoked",
            code="lease_revoked",
        )
    if live.expires_at_ms <= observed:
        raise QuackOwnerMutationEnvelopeError(
            "authorized owner mutation lease is expired",
            code="lease_expired",
        )
    if (
        requested.fence_epoch != live.fence_epoch
        or requested.fencing_token != live.fencing_token
    ):
        raise FenceMismatchError(
            "stale fence cannot authorize an owner mutation",
            details={
                "requested_fence_epoch": requested.fence_epoch,
                "live_fence_epoch": live.fence_epoch,
                "requested_fencing_token": requested.fencing_token,
                "live_fencing_token": live.fencing_token,
            },
        )
    return live


def assert_mutation_idempotency(
    requested: OwnerMutationAuthority,
    existing: Mapping[str, Any] | None,
    *,
    worker_assertion: bool = False,
) -> Mapping[str, Any] | None:
    """Return the prior result for an exact digest replay, or None to admit.

    A reused idempotency key with a different mutation payload fails closed.
    Worker assertions cannot coerce a conflict into a replay or a new effect.
    """

    if not isinstance(requested, OwnerMutationAuthority):
        raise ControlPlaneContractError(
            "idempotency admission requires OwnerMutationAuthority"
        )
    _ = worker_assertion
    if existing is None:
        return None
    if not isinstance(existing, Mapping):
        raise QuackOwnerMutationEnvelopeError(
            "idempotency record must be an object",
            code="idempotency_conflict",
        )
    existing_key = str(existing.get("idempotency_key") or "").strip()
    if existing_key and existing_key != requested.idempotency_key:
        raise IdempotencyConflictError(
            "idempotency record does not match the requested key",
            details={
                "requested_idempotency_key": requested.idempotency_key,
                "existing_idempotency_key": existing_key,
            },
        )
    existing_digest = _existing_idempotency_digest(existing)
    if existing_digest != requested.request_digest:
        raise IdempotencyConflictError(
            "idempotency key already bound to a different mutation payload",
            details={
                "idempotency_key": requested.idempotency_key,
                "requested_digest": requested.request_digest,
                "existing_digest": existing_digest,
            },
        )
    return _existing_idempotency_result(existing)


def admit_owner_mutation(
    requested: OwnerMutationAuthority,
    live: LiveOwnerClaimLease,
    *,
    now_ms: int,
    existing_idempotency: Mapping[str, Any] | None = None,
    expected_task: TaskStateSnapshot | None = None,
    proposed_task: TaskStateSnapshot | None = None,
    current_task: TaskStateSnapshot | None = None,
    worker_assertion: bool = False,
) -> OwnerMutationAdmission:
    """Admit one owner mutation under live claim, lease, fence, and idempotency.

    Envelope authentication remains ``parse_mutation_request``.  This gate is
    the exclusive-owner authority check that must precede SQL execution.
    ``CanonicalTaskStateMachine@1`` and ``RevisionCASTransition@1`` are
    consumed when task snapshots are supplied; they are not reimplemented.
    """

    assert_live_claim_lease_fence(
        requested,
        live,
        now_ms=now_ms,
        worker_assertion=worker_assertion,
    )
    replay = assert_mutation_idempotency(
        requested,
        existing_idempotency,
        worker_assertion=worker_assertion,
    )
    if replay is not None:
        return OwnerMutationAdmission(
            outcome=CommandOutcome.IDEMPOTENT_REPLAY,
            changed=False,
            requested=requested,
            live=live,
            replay_result=replay,
        )
    if expected_task is not None or proposed_task is not None:
        if not isinstance(expected_task, TaskStateSnapshot) or not isinstance(
            proposed_task, TaskStateSnapshot
        ):
            raise ControlPlaneContractError(
                "task-bound owner mutations require expected and proposed snapshots"
            )
        if (
            expected_task.task_cid != requested.task_cid
            or proposed_task.task_cid != requested.task_cid
        ):
            raise QuackOwnerMutationEnvelopeError(
                "task snapshot identity does not match the mutation claim",
                code="claim_mismatch",
            )
        if expected_task.lease_id != requested.lease_id:
            raise QuackOwnerMutationEnvelopeError(
                "task snapshot lease does not match the owner mutation binding",
                code="lease_mismatch",
            )
        if expected_task.fence_epoch != requested.fence_epoch:
            raise FenceMismatchError(
                "task snapshot fence does not match the owner mutation binding",
                details={
                    "snapshot_fence_epoch": expected_task.fence_epoch,
                    "requested_fence_epoch": requested.fence_epoch,
                },
            )
        assert_task_cas_transition(
            expected_task,
            proposed_task,
            current=current_task,
            worker_assertion=worker_assertion,
        )
        if proposed_task.state is TaskState.COMPLETED:
            current = expected_task if current_task is None else current_task
            if not expected_task.may_complete_against(current):
                raise ControlPlaneContractError(
                    "worker or model assertion cannot complete a task without current claim, lease and fence"
                )
    return OwnerMutationAdmission(
        outcome=CommandOutcome.ACCEPTED,
        changed=True,
        requested=requested,
        live=live,
    )


def claim_lease_fence_idempotency_allowed(
    requested: OwnerMutationAuthority,
    live: LiveOwnerClaimLease,
    *,
    now_ms: int,
    existing_idempotency: Mapping[str, Any] | None = None,
    expected_task: TaskStateSnapshot | None = None,
    proposed_task: TaskStateSnapshot | None = None,
    current_task: TaskStateSnapshot | None = None,
    worker_assertion: bool = False,
) -> bool:
    """Return whether owner-mutation admission would succeed without raising."""

    try:
        admit_owner_mutation(
            requested,
            live,
            now_ms=now_ms,
            existing_idempotency=existing_idempotency,
            expected_task=expected_task,
            proposed_task=proposed_task,
            current_task=current_task,
            worker_assertion=worker_assertion,
        )
    except (
        QuackOwnerMutationEnvelopeError,
        ControlPlaneContractError,
        TransactionError,
    ):
        return False
    return True


def read_envelope(path: Path, *, max_bytes: int = MAX_MUTATION_RESULT_BYTES) -> Mapping[str, Any]:
    target = Path(path)
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target, flags)
        with os.fdopen(descriptor, "rb") as stream:
            observed = os.fstat(stream.fileno())
            if (
                not stat_module.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
                or observed.st_uid != os.geteuid()
                or stat_module.S_IMODE(observed.st_mode) & 0o077
                or observed.st_size < 2
                or observed.st_size > max_bytes
            ):
                raise QuackOwnerMutationEnvelopeError(
                    "mutation envelope file is unbounded or unsafe"
                )
            encoded = stream.read(max_bytes + 1)
            if len(encoded) != observed.st_size:
                raise QuackOwnerMutationEnvelopeError("mutation envelope changed while being read")
        payload = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QuackOwnerMutationEnvelopeError("mutation envelope JSON is malformed") from exc
    if not isinstance(payload, Mapping):
        raise QuackOwnerMutationEnvelopeError("mutation envelope must contain an object")
    return MappingProxyType(dict(payload))


__all__ = (
    "ACCEPTED_LEASE_STATE",
    "CLAIM_LEASE_FENCE_IDEMPOTENCY_BINDING",
    "CLAIM_LEASE_FENCE_IDEMPOTENCY_SCHEMA",
    "LIVE_OWNER_CLAIM_LEASE_SCHEMA",
    "MAX_MUTATION_REQUEST_AGE_MS",
    "MAX_MUTATION_REQUEST_BYTES",
    "MAX_MUTATION_RESULT_BYTES",
    "MAX_MUTATION_RESULT_ROWS",
    "MAX_MUTATION_SQL_BYTES",
    "OWNER_MUTATION_AUTHORITY_SCHEMA",
    "QUACK_OWNER_MUTATION_INTERFACE",
    "QUACK_OWNER_MUTATION_REQUEST_SCHEMA",
    "QUACK_OWNER_MUTATION_RESULT_SCHEMA",
    "LiveOwnerClaimLease",
    "OwnerMutationAdmission",
    "OwnerMutationAuthority",
    "QuackOwnerMutationEnvelopeError",
    "admit_mutation_sql",
    "admit_owner_mutation",
    "assert_live_claim_lease_fence",
    "assert_mutation_idempotency",
    "build_mutation_request",
    "build_mutation_result",
    "claim_lease_fence_idempotency_allowed",
    "mutation_envelope_exists_at",
    "mutation_request_digest",
    "open_mutation_inbox_directory",
    "parse_mutation_request",
    "parse_mutation_result",
    "read_envelope",
    "read_envelope_at",
    "unlink_mutation_envelope_at",
    "write_envelope_atomic",
    "write_envelope_atomic_at",
)
