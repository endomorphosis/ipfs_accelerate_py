"""Authenticated, bounded envelopes for Quack owner-side mutations.

The Quack beta transport can read and insert attached rows, but some builds
cannot update or delete base tables.  Those operations are handed to the
exclusive state owner through a local, mode-0700 inbox.  This module keeps the
handoff typed and authenticated without ever writing the raw Quack token.
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
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

QUACK_OWNER_MUTATION_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-owner-mutation-request@1"
)
QUACK_OWNER_MUTATION_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-owner-mutation-result@1"
)

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
    "MAX_MUTATION_REQUEST_AGE_MS",
    "MAX_MUTATION_REQUEST_BYTES",
    "MAX_MUTATION_RESULT_BYTES",
    "MAX_MUTATION_RESULT_ROWS",
    "MAX_MUTATION_SQL_BYTES",
    "QUACK_OWNER_MUTATION_REQUEST_SCHEMA",
    "QUACK_OWNER_MUTATION_RESULT_SCHEMA",
    "QuackOwnerMutationEnvelopeError",
    "admit_mutation_sql",
    "build_mutation_request",
    "build_mutation_result",
    "parse_mutation_request",
    "parse_mutation_result",
    "read_envelope",
    "write_envelope_atomic",
)
